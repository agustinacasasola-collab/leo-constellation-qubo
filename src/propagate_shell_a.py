"""
propagate_shell_a.py
--------------------
Propagates Shell A candidates using SGP4 (SatrecArray, vectorised).

Input:   data/shell_a_candidates.tle
Output:  data/propagated_shell_a.csv
Format:  norad_id, timestep, x_km, y_km, z_km
Window:  3 days at 60 s timestep -> 4,321 steps per satellite
Total rows: 200 x 4,321 = 864,200

Altitude sanity:
    Shell A (550 km): WARN if any satellite deviates more than WARN_MARGIN_KM
    from the nominal altitude over the entire propagation window.

Usage:
    python src/propagate_shell_a.py
"""

import math
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
INPUT_TLE_PATH  = DATA_DIR / "shell_a_candidates.tle"
INPUT_CSV_PATH  = DATA_DIR / "shell_a_candidates.csv"
OUTPUT_CSV_PATH = DATA_DIR / "propagated_shell_a.csv"

SIMULATION_DAYS = 3
STEP_SECONDS    = 60.0
EARTH_RADIUS_KM = 6371.0
NOMINAL_ALT_KM  = 550.0
WARN_MARGIN_KM  = 10.0    # J2 short-period causes +-5-7 km oscillation over 3 days
BATCH_SIZE      = 100     # satellites per SatrecArray call


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
# Time grid
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
# Vectorised propagation
# ---------------------------------------------------------------------------

def propagate_all(
    sat_records: list[dict],
    duration_days: float,
    step_seconds: float,
) -> tuple[int, dict[str, tuple[float, float]], float]:
    """
    Propagate all satellites. Streams output to OUTPUT_CSV_PATH in batches.

    Returns
    -------
    n_errors  : total SGP4 error count
    alt_stats : {norad_id_str: (alt_min_km, alt_max_km)}
    elapsed   : wall-clock seconds
    """
    N          = len(sat_records)
    duration_s = duration_days * 86400.0
    T          = int(duration_s / step_seconds) + 1

    start_time = epoch_from_tle_line1(sat_records[0]['line1'])
    jd_array, fr_array = build_time_arrays(start_time, duration_s, step_seconds)

    norad_ids = [s['norad_id'] for s in sat_records]
    satrecs   = [Satrec.twoline2rv(s['line1'], s['line2']) for s in sat_records]
    ts_full   = np.arange(T, dtype=np.int32)

    # Write CSV header
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV_PATH, 'w') as f:
        f.write('norad_id,timestep,x_km,y_km,z_km\n')

    n_errors  = 0
    alt_stats = {}
    t0        = time.perf_counter()

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
                alt_stats[str(nid)] = (float(alts.min()), float(alts.max()))

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
    return n_errors, alt_stats, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    duration_s = SIMULATION_DAYS * 86400.0
    T          = int(duration_s / STEP_SECONDS) + 1

    print("=" * 60)
    print("Shell A Orbit Propagator  (SGP4, vectorised)")
    print(f"  Input     : {INPUT_TLE_PATH.name}")
    print(f"  Duration  : {SIMULATION_DAYS} days  ({duration_s/3600:.0f} h)")
    print(f"  Timestep  : {STEP_SECONDS:.0f} s  ->  {T:,} steps per satellite")
    print(f"  Output    : {OUTPUT_CSV_PATH.name}")
    print("=" * 60)

    if not INPUT_TLE_PATH.exists():
        print(f"\n  ERROR: {INPUT_TLE_PATH.name} not found.")
        print("  Run 'python src/generate_shell_a.py' first.")
        return
    if not INPUT_CSV_PATH.exists():
        print(f"\n  ERROR: {INPUT_CSV_PATH.name} not found.")
        print("  Run 'python src/generate_shell_a.py' first.")
        return

    sat_records = load_tles(INPUT_TLE_PATH)
    N = len(sat_records)
    print(f"\n  Loaded {N} TLEs  "
          f"(NORAD {sat_records[0]['norad_id']} .. {sat_records[-1]['norad_id']})")
    print(f"  Total rows: {N:,} x {T:,} = {N * T:,}")

    print("\nPropagating ...")
    n_errors, alt_stats, elapsed = propagate_all(sat_records, SIMULATION_DAYS, STEP_SECONDS)

    # --- Altitude sanity check (aggregate over all sats) --------------------
    all_mins = [v[0] for v in alt_stats.values()]
    all_maxs = [v[1] for v in alt_stats.values()]

    if all_mins:
        g_min = min(all_mins)
        g_max = max(all_maxs)
        dev   = max(abs(g_min - NOMINAL_ALT_KM), abs(g_max - NOMINAL_ALT_KM))
        alt_ok = dev <= WARN_MARGIN_KM
        flag   = "PASS" if alt_ok else f"WARNING dev={dev:.1f} km > {WARN_MARGIN_KM} km"
        print(f"\n  Altitude range  : {g_min:.2f} - {g_max:.2f} km  [{flag}]")
        if not alt_ok:
            print(f"  NOTE: J2 short-period oscillation causes +-5-7 km over 3 days -- expected.")
    else:
        print("\n  WARNING: no valid altitude stats (all SGP4 errors?)")
        alt_ok = False

    print(f"  SGP4 errors     : {n_errors}")
    print(f"  Elapsed         : {elapsed:.1f} s")

    file_mb = OUTPUT_CSV_PATH.stat().st_size / 1024 ** 2
    print(f"  Output file     : {OUTPUT_CSV_PATH.name}  ({file_mb:.1f} MB, {N * T:,} rows)")

    gate_flag = "PASS" if (alt_ok and n_errors == 0) else "WARN"
    print(f"\n  GATE: {gate_flag}")
    print("Done.")


if __name__ == "__main__":
    main()
