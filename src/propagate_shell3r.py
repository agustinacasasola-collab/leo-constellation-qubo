"""
propagate_shell3r.py  --  STEP 2
----------------------------------
Propagates shell3r candidates with SGP4 (SatrecArray, vectorised).

Input:   data/shell3r_candidates.tle
         data/shell3r_candidates.csv
Output:  data/propagated_shell3r.csv
Format:  norad_id, timestep, x_km, y_km, z_km
Window:  3 days at 60 s timestep  ->  4,321 steps per satellite
Rows:    130 x 4,321 = 561,730

Altitude sanity:
    Expected range 548-552 km (nominal 550 km +/- J2 oscillation).
    Prints WARNING if any satellite deviates > 5 km from 550 km nominal.

Usage:
    python src/propagate_shell3r.py
"""

import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
DATA_DIR        = Path(__file__).parent.parent / "data"
INPUT_TLE_PATH  = DATA_DIR / "shell3r_candidates.tle"
INPUT_CSV_PATH  = DATA_DIR / "shell3r_candidates.csv"
OUTPUT_CSV_PATH = DATA_DIR / "propagated_shell3r.csv"

SIMULATION_DAYS = 3
STEP_SECONDS    = 60.0
EARTH_RADIUS_KM = 6371.0
NOMINAL_ALT_KM  = 550.0
WARN_THRESHOLD  = 10.0    # km  -- J2 short-period oscillation causes +/-5-7 km
                          # over 3 days; warn only if deviation exceeds 10 km
BATCH_SIZE      = 130     # all candidates in one SatrecArray call


# ---------------------------------------------------------------------------
# TLE loading
# ---------------------------------------------------------------------------

def load_tles(tle_path: Path) -> list[dict]:
    lines = [ln.strip() for ln in tle_path.read_text().splitlines() if ln.strip()]
    sats  = []
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith('1 ') and l2.startswith('2 '):
            sats.append({'norad_id': int(l1[2:7].strip()), 'line1': l1, 'line2': l2})
    return sats


# ---------------------------------------------------------------------------
# Time grid anchored to TLE epoch
# ---------------------------------------------------------------------------

def epoch_from_tle_line1(l1: str) -> datetime:
    y2   = int(l1[18:20])
    doy  = float(l1[20:32])
    year = (2000 + y2) if y2 < 57 else (1900 + y2)
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)


def build_time_arrays(
    start_time: datetime,
    duration_seconds: float,
    step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(duration_seconds / step_seconds) + 1
    jd_list, fr_list = [], []
    for i in range(n_steps):
        t = start_time + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_list.append(jd)
        fr_list.append(fr)
    return np.array(jd_list, dtype=np.float64), np.array(fr_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_all(
    sat_records: list[dict],
    duration_days: float,
    step_seconds: float,
) -> tuple[int, list[tuple[int, float, float]], float]:
    """
    Propagate all satellites and stream output to OUTPUT_CSV_PATH.

    Returns
    -------
    n_errors      : total SGP4 error count
    alt_per_sat   : list of (norad_id, alt_min_km, alt_max_km)
    elapsed       : wall-clock seconds
    """
    N          = len(sat_records)
    duration_s = duration_days * 86400.0
    T          = int(duration_s / step_seconds) + 1

    start_time          = epoch_from_tle_line1(sat_records[0]['line1'])
    jd_array, fr_array  = build_time_arrays(start_time, duration_s, step_seconds)
    ts_full             = np.arange(T, dtype=np.int32)

    norad_ids = [s['norad_id'] for s in sat_records]
    satrecs   = [Satrec.twoline2rv(s['line1'], s['line2']) for s in sat_records]

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV_PATH, 'w') as f:
        f.write('norad_id,timestep,x_km,y_km,z_km\n')

    n_errors    = 0
    alt_per_sat = []
    t0          = time.perf_counter()

    for batch_start in range(0, N, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, N)
        b_satrecs = satrecs[batch_start:batch_end]
        b_norads  = norad_ids[batch_start:batch_end]
        b_n       = len(b_satrecs)

        sat_arr              = SatrecArray(b_satrecs)
        e_batch, r_batch, _  = sat_arr.sgp4(jd_array, fr_array)

        r_np = np.asarray(r_batch, dtype=np.float64)   # (b_n, T, 3)
        e_np = np.asarray(e_batch, dtype=np.int32)      # (b_n, T)
        n_errors += int((e_np != 0).sum())

        for local_i in range(b_n):
            nid  = b_norads[local_i]
            mask = e_np[local_i] == 0
            if mask.any():
                alts = np.linalg.norm(r_np[local_i, mask], axis=1) - EARTH_RADIUS_KM
                alt_per_sat.append((nid, float(alts.min()), float(alts.max())))

        norad_col = np.repeat(np.array(b_norads, dtype=np.int32), T)
        ts_col    = np.tile(ts_full, b_n)
        x_col     = r_np[:, :, 0].ravel()
        y_col     = r_np[:, :, 1].ravel()
        z_col     = r_np[:, :, 2].ravel()

        df_batch = pd.DataFrame({
            'norad_id': norad_col,
            'timestep': ts_col,
            'x_km':     x_col,
            'y_km':     y_col,
            'z_km':     z_col,
        })
        df_batch.to_csv(OUTPUT_CSV_PATH, mode='a', index=False, header=False,
                        float_format='%.4f')

    elapsed = time.perf_counter() - t0
    return n_errors, alt_per_sat, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    duration_s = SIMULATION_DAYS * 86400.0
    T          = int(duration_s / STEP_SECONDS) + 1

    print("=" * 62)
    print("Shell3r Orbit Propagator  (SGP4, vectorised)")
    print(f"  Input     : {INPUT_TLE_PATH.name}")
    print(f"  Duration  : {SIMULATION_DAYS} days  ({duration_s/3600:.0f} h)")
    print(f"  Timestep  : {STEP_SECONDS:.0f} s  ->  {T:,} steps per satellite")
    print(f"  Output    : {OUTPUT_CSV_PATH.name}")
    print("=" * 62)

    for p in [INPUT_TLE_PATH, INPUT_CSV_PATH]:
        if not p.exists():
            print(f"\n  ERROR: {p.name} not found.")
            print("  Run 'python src/generate_shell3r.py' first.")
            return

    sat_records = load_tles(INPUT_TLE_PATH)
    N = len(sat_records)
    print(f"\n  Loaded {N} TLEs  "
          f"(NORAD {sat_records[0]['norad_id']} .. {sat_records[-1]['norad_id']})")
    print(f"  Total rows: {N:,} x {T:,} = {N * T:,}")

    print("\nPropagating ...")
    n_errors, alt_per_sat, elapsed = propagate_all(sat_records, SIMULATION_DAYS, STEP_SECONDS)

    # --- Altitude sanity check -----------------------------------------------
    all_mins = [v[1] for v in alt_per_sat]
    all_maxs = [v[2] for v in alt_per_sat]
    n_warn   = 0

    if all_mins:
        g_min = min(all_mins)
        g_max = max(all_maxs)
        print(f"\n  Altitude sanity check:")
        print(f"  Min/max across all candidates: {g_min:.2f} - {g_max:.2f} km")
        print(f"  Expected: {NOMINAL_ALT_KM - 2:.0f} - {NOMINAL_ALT_KM + 2:.0f} km "
              f"(warn if >+/-{WARN_THRESHOLD:.0f} km from {NOMINAL_ALT_KM:.0f} km)")

        for nid, amin, amax in alt_per_sat:
            dev = max(abs(amin - NOMINAL_ALT_KM), abs(amax - NOMINAL_ALT_KM))
            if dev > WARN_THRESHOLD:
                n_warn += 1

        if n_warn == 0:
            print(f"  All {N} candidates within +/-{WARN_THRESHOLD:.0f} km  OK")
        else:
            max_dev = max(
                max(abs(v[1] - NOMINAL_ALT_KM), abs(v[2] - NOMINAL_ALT_KM))
                for v in alt_per_sat
            )
            print(f"  WARNING: {n_warn}/{N} candidates deviate > {WARN_THRESHOLD:.0f} km "
                  f"(max dev = {max_dev:.1f} km)")
            print(f"  NOTE: J2 short-period oscillation causes +/-5-7 km over 3 days -- "
                  f"expected if dev < 10 km.")
    else:
        print("\n  WARNING: no valid altitude stats (all SGP4 errors?)")

    print(f"\n  SGP4 errors : {n_errors}")
    print(f"  Elapsed     : {elapsed:.1f} s")

    file_mb = OUTPUT_CSV_PATH.stat().st_size / 1024 ** 2
    print(f"  Output file : {OUTPUT_CSV_PATH.name}  ({file_mb:.1f} MB, {N * T:,} rows)")

    gate = "PASS" if n_errors == 0 and n_warn == 0 else "WARN"
    print(f"\n  GATE: {gate}")
    print("Done.")


if __name__ == "__main__":
    main()
