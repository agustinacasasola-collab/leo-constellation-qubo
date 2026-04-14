"""
propagate_multishell.py
-----------------------
Propagates multishell candidates using SGP4 (SatrecArray, vectorised).
Same logic as propagate_orbits.py --synthetic mode.

Input:   data/multishell_candidates.tle
Output:  data/propagated_multishell.csv
Format:  norad_id, timestep, x_km, y_km, z_km
Window:  3 days at 60 s timestep  →  4,321 steps per satellite

Altitude sanity check after propagation (per shell):
    Shell A (550 km): expect 548 – 552 km
    Shell B (560 km): expect 558 – 562 km
    Shell C (560 km): expect 558 – 562 km
    WARNING printed if any shell deviates more than 5 km.

Usage:
    python src/propagate_multishell.py
"""

import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # type: ignore[misc]
        return it

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
DATA_DIR          = Path(__file__).parent.parent / "data"
INPUT_TLE_PATH    = DATA_DIR / "multishell_candidates.tle"
INPUT_CSV_PATH    = DATA_DIR / "multishell_candidates.csv"
OUTPUT_CSV_PATH   = DATA_DIR / "propagated_multishell.csv"

SIMULATION_DAYS   = 3
STEP_SECONDS      = 60.0
EARTH_RADIUS_KM   = 6371.0
BATCH_SIZE        = 100   # satellites per SatrecArray call (18 total, so one batch)

# Expected altitude ranges per shell (nominal ± 2 km, warn at ± 5 km)
SHELL_EXPECTED = {
    "A": (550.0, 2.0),   # (nominal_km, tight_margin_km)
    "B": (560.0, 2.0),
    "C": (560.0, 2.0),
}
WARN_MARGIN_KM = 10.0  # J2 short-period term causes +-5-7 km altitude oscillation over 3 days


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
# Time grid anchored to TLE epoch (matches propagated_catalog.csv indexing)
# ---------------------------------------------------------------------------

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


def epoch_from_tle_line1(l1: str) -> datetime:
    y2  = int(l1[18:20])
    doy = float(l1[20:32])
    year = (2000 + y2) if y2 < 57 else (1900 + y2)
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)


# ---------------------------------------------------------------------------
# Vectorised propagation (mirrors propagate_orbits.py propagate_synthetic_vectorized)
# ---------------------------------------------------------------------------

def propagate_all(
    sat_records: list[dict],
    duration_days: float,
    step_seconds: float,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, dict[str, tuple[float, float]], float]:
    """
    Propagate all satellites with SatrecArray and stream to OUTPUT_CSV_PATH.

    Returns
    -------
    n_errors  : int
    alt_stats : dict {norad_id_str: (alt_min, alt_max)}
    elapsed   : float  (wall-clock seconds)
    """
    N          = len(sat_records)
    duration_s = duration_days * 86400.0
    T          = int(duration_s / step_seconds) + 1

    # Anchor to the first TLE's epoch (same as propagate_orbits.py --synthetic)
    start_time = epoch_from_tle_line1(sat_records[0]['line1'])
    jd_array, fr_array = build_time_arrays(start_time, duration_s, step_seconds)

    # Pre-parse Satrec objects
    norad_ids = [s['norad_id'] for s in sat_records]
    satrecs   = [Satrec.twoline2rv(s['line1'], s['line2']) for s in sat_records]
    ts_full   = np.arange(T, dtype=np.int32)

    # Write CSV header
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV_PATH, 'w') as f:
        f.write('norad_id,timestep,x_km,y_km,z_km\n')

    n_errors  = 0
    alt_stats = {}   # norad_id_str → (alt_min, alt_max)
    t0        = time.perf_counter()

    for batch_start in range(0, N, batch_size):
        batch_end  = min(batch_start + batch_size, N)
        b_satrecs  = satrecs[batch_start:batch_end]
        b_norads   = norad_ids[batch_start:batch_end]
        b_n        = len(b_satrecs)

        sat_arr            = SatrecArray(b_satrecs)
        e_batch, r_batch, _ = sat_arr.sgp4(jd_array, fr_array)

        r_np = np.asarray(r_batch, dtype=np.float64)   # (b_n, T, 3)
        e_np = np.asarray(e_batch, dtype=np.int32)     # (b_n, T)
        n_errors += int((e_np != 0).sum())

        # Altitude stats per satellite
        for local_i in range(b_n):
            nid  = b_norads[local_i]
            mask = e_np[local_i] == 0
            if mask.any():
                alts = np.linalg.norm(r_np[local_i, mask], axis=1) - EARTH_RADIUS_KM
                alt_stats[str(nid)] = (float(alts.min()), float(alts.max()))

        # Build output DataFrame (no Python loops)
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

    print("=" * 65)
    print("Multi-Shell Orbit Propagator")
    print(f"  Input     : {INPUT_TLE_PATH.name}")
    print(f"  Duration  : {SIMULATION_DAYS} days  ({duration_s/3600:.0f} h)")
    print(f"  Timestep  : {STEP_SECONDS:.0f} s  ->  {T:,} steps per satellite")
    print(f"  Output    : {OUTPUT_CSV_PATH.name}")
    print("=" * 65)

    # --- Prerequisite checks -----------------------------------------------
    if not INPUT_TLE_PATH.exists():
        print(f"\n  ERROR: {INPUT_TLE_PATH} not found.")
        print("  Run 'python src/generate_multishell_candidates.py' first.")
        return
    if not INPUT_CSV_PATH.exists():
        print(f"\n  ERROR: {INPUT_CSV_PATH} not found.")
        print("  Run 'python src/generate_multishell_candidates.py' first.")
        return

    # --- Load TLEs and candidate metadata ----------------------------------
    sat_records = load_tles(INPUT_TLE_PATH)
    N = len(sat_records)
    print(f"\n  Loaded {N} TLEs  "
          f"(NORAD {sat_records[0]['norad_id']} .. {sat_records[-1]['norad_id']})")
    print(f"  Total rows: {N:,} x {T:,} = {N*T:,}")

    df_meta = pd.read_csv(INPUT_CSV_PATH)
    # Build lookup: norad_id → shell_label
    id_to_shell = dict(zip(df_meta['norad_id'].astype(str), df_meta['shell_label']))

    # --- Propagate ----------------------------------------------------------
    print("\nPropagating...")
    n_errors, alt_stats, elapsed = propagate_all(sat_records, SIMULATION_DAYS, STEP_SECONDS)

    # --- Altitude sanity check per shell ------------------------------------
    print()
    print("=" * 65)
    print("ALTITUDE SANITY CHECK")
    print("=" * 65)
    print(f"  {'Shell':>7}  {'NORAD':>7}  {'AltMin':>8}  {'AltMax':>8}  {'Status'}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*20}")

    gate_ok = True
    shell_alts: dict[str, list[tuple[float, float]]] = {"A": [], "B": [], "C": []}

    for s in sat_records:
        nid   = str(s['norad_id'])
        label = id_to_shell.get(nid, "?")
        if nid not in alt_stats:
            print(f"  Shell {label}  {nid:>7}  {'ERR':>8}  {'ERR':>8}  SGP4 failure")
            continue
        alt_min, alt_max = alt_stats[nid]
        if label in shell_alts:
            shell_alts[label].append((alt_min, alt_max))
        nominal, _ = SHELL_EXPECTED.get(label, (555.0, 2.0))
        dev = max(abs(alt_min - nominal), abs(alt_max - nominal))
        status = "OK" if dev <= WARN_MARGIN_KM else f"WARNING (dev={dev:.1f} km)"
        print(f"  Shell {label}  {nid:>7}  {alt_min:>8.2f}  {alt_max:>8.2f}  {status}")

    # Per-shell summary
    print()
    all_pass = True
    for label in ("A", "B", "C"):
        if not shell_alts[label]:
            continue
        all_mins = [v[0] for v in shell_alts[label]]
        all_maxs = [v[1] for v in shell_alts[label]]
        g_min = min(all_mins)
        g_max = max(all_maxs)
        nominal, _ = SHELL_EXPECTED[label]
        dev = max(abs(g_min - nominal), abs(g_max - nominal))
        inc = df_meta[df_meta['shell_label'] == label]['inc_deg'].iloc[0]
        ok  = dev <= WARN_MARGIN_KM
        all_pass = all_pass and ok
        flag = "PASS" if ok else f"WARNING dev={dev:.1f} km > {WARN_MARGIN_KM} km"
        print(f"  Shell {label} ({nominal:.0f} km, {inc:.1f}deg): "
              f"alt range = {g_min:.2f} – {g_max:.2f} km  [{flag}]")

    if not all_pass:
        print("\n  WARNING: One or more shells deviate more than "
              f"{WARN_MARGIN_KM} km from expected altitude.")

    # --- Gate: altitude sanity -----------------------------------------------
    print()
    print(f"  GATE: {'PASS' if all_pass else 'WARN'} - altitude sanity check")
    print(f"  SGP4 errors: {n_errors}")
    print(f"  Computation : {elapsed:.1f} s")
    file_mb = OUTPUT_CSV_PATH.stat().st_size / 1024 ** 2
    print(f"  Output file : {OUTPUT_CSV_PATH.name}  ({file_mb:.1f} MB, {N*T:,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
