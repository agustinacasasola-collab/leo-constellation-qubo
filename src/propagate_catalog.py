"""
propagate_catalog.py
--------------------
Pre-computes GCRF positions for all LEO catalog objects over a 3-day
simulation window at 60 s timesteps, anchored to the Shell 3 TLE epoch.

Output
------
data/propagated_catalog.csv
    Columns : norad_id (int32), timestep (int16), x_km, y_km, z_km  (float32)
    Rows    : n_catalog × 4,321
    Size    : ~25,928 × 4,321 × (4+2+12) bytes ≈ 1.5 GB (float32)

Float32 is used to keep the file and in-memory footprint manageable;
position precision is ~3 m, more than adequate for 10 km screening.

Strategy
--------
SatrecArray propagates each batch of satellites over all T timesteps in
one vectorised NumPy call.  Only rows where the SGP4 error code == 0
are written; SGP4-failed rows are omitted entirely (the downstream code
treats absent rows as invalid).

To reduce memory, results are streamed to CSV in batches.

Usage
-----
    python src/propagate_catalog.py
    python src/propagate_catalog.py --batch-size 1000
"""

import argparse
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR          = Path(__file__).parent.parent
DATA_DIR          = ROOT_DIR / "data"
CATALOG_TLE_PATH  = DATA_DIR / "leo_catalog.tle"
SYNTHETIC_TLE_PATH = DATA_DIR / "shell3_synthetic.tle"
OUTPUT_CSV        = DATA_DIR / "propagated_catalog.csv"

# ---------------------------------------------------------------------------
# Simulation parameters  (must match compute_pc.py / propagate_orbits.py)
# ---------------------------------------------------------------------------
SIMULATION_DAYS = 3
STEP_SECONDS    = 60.0
DEFAULT_BATCH   = 500
PRINT_EVERY     = 1_000   # print progress every N objects

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tle_pairs(path: Path) -> list[tuple[str, str]]:
    lines = [l.rstrip() for l in open(path) if l.strip()]
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def parse_tle_epoch(line1: str) -> datetime:
    """Parse TLE Line 1 epoch (cols 18-32) → UTC datetime."""
    y2      = int(line1[18:20])
    doy_frac = float(line1[20:32])
    year    = (2000 + y2) if y2 < 57 else (1900 + y2)
    jan1    = datetime(year, 1, 1, tzinfo=timezone.utc)
    return jan1 + timedelta(days=doy_frac - 1.0)


def build_time_arrays(
    start_dt: datetime,
    n_steps: int,
    step_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Julian-date arrays (jd, fr) for all timesteps.

    Each (jd[t], fr[t]) is computed individually via jday() so that
    fr is always in [0, 1) — compatible with SatrecArray.sgp4.

    Returns
    -------
    jd_array : ndarray float64, shape (n_steps,)
    fr_array : ndarray float64, shape (n_steps,)
    """
    jd_list, fr_list = [], []
    for i in range(n_steps):
        t  = start_dt + timedelta(seconds=i * step_s)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_list.append(jd)
        fr_list.append(fr)
    return np.array(jd_list, dtype=np.float64), np.array(fr_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-propagate LEO catalog over 3-day window."
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"Satellites per SatrecArray call (default: {DEFAULT_BATCH})"
    )
    args = parser.parse_args()

    t_wall = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load TLE files
    # ------------------------------------------------------------------
    if not CATALOG_TLE_PATH.exists():
        print(f"ERROR: {CATALOG_TLE_PATH} not found.")
        return
    if not SYNTHETIC_TLE_PATH.exists():
        print(f"ERROR: {SYNTHETIC_TLE_PATH} not found.  "
              "Run generate_candidates.py first.")
        return

    cat_pairs = load_tle_pairs(CATALOG_TLE_PATH)
    syn_pairs = load_tle_pairs(SYNTHETIC_TLE_PATH)
    N_cat = len(cat_pairs)

    # ------------------------------------------------------------------
    # 2. Derive simulation epoch from Shell 3 TLE (same as compute_pc.py)
    # ------------------------------------------------------------------
    epoch_dt = parse_tle_epoch(syn_pairs[0][0])
    T        = int(SIMULATION_DAYS * 86400 / STEP_SECONDS) + 1   # 4,321

    print("=" * 70)
    print("propagate_catalog.py — LEO catalog pre-propagation")
    print(f"  Catalog TLE     : {CATALOG_TLE_PATH.name}  ({N_cat:,} objects)")
    print(f"  Epoch (Shell 3) : {epoch_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Duration        : {SIMULATION_DAYS} days  "
          f"({T:,} timesteps @ {STEP_SECONDS:.0f} s)")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Output          : {OUTPUT_CSV.name}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 3. Build time arrays once
    # ------------------------------------------------------------------
    print(f"\nBuilding {T:,} Julian-date timesteps ...")
    jd_arr, fr_arr = build_time_arrays(epoch_dt, T, STEP_SECONDS)
    print(f"  fr range: {fr_arr.min():.6f} .. {fr_arr.max():.6f}  "
          "(all in [0, 1) — SatrecArray-safe)")

    # ------------------------------------------------------------------
    # 4. Stream-write CSV in batches
    # ------------------------------------------------------------------
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w") as fout:
        fout.write("norad_id,timestep,x_km,y_km,z_km\n")

    n_written  = 0
    n_errors   = 0
    n_objects  = 0
    batch_size = args.batch_size

    print(f"\nPropagating {N_cat:,} catalog objects ...")
    t_start = time.perf_counter()

    for batch_start in range(0, N_cat, batch_size):
        batch_end   = min(batch_start + batch_size, N_cat)
        batch_pairs = cat_pairs[batch_start:batch_end]
        b_n         = len(batch_pairs)

        # Parse TLEs — skip unparseable ones
        norad_ids  = []
        valid_sats = []
        for l1, l2 in batch_pairs:
            try:
                sat = Satrec.twoline2rv(l1, l2)
                valid_sats.append(sat)
                norad_ids.append(int(l1[2:7].strip()))
            except Exception:
                n_errors += 1

        if not valid_sats:
            continue

        b_valid = len(valid_sats)
        sa      = SatrecArray(valid_sats)

        # Vectorised propagation: e (b_valid, T), r (b_valid, T, 3)
        e_batch, r_batch, _ = sa.sgp4(jd_arr, fr_arr)

        e_np = np.asarray(e_batch, dtype=np.int8)     # (b_valid, T)
        r_np = np.asarray(r_batch, dtype=np.float32)  # (b_valid, T, 3)

        n_errors += int((e_np != 0).sum())

        # Build output DataFrame — only valid (error=0) rows
        sat_idx, ts_idx = np.where(e_np == 0)          # (n_valid_rows,)

        if len(sat_idx) == 0:
            n_objects += b_valid
            continue

        norad_col = np.array(norad_ids, dtype=np.int32)[sat_idx]
        ts_col    = ts_idx.astype(np.int16)
        x_col     = r_np[sat_idx, ts_idx, 0]
        y_col     = r_np[sat_idx, ts_idx, 1]
        z_col     = r_np[sat_idx, ts_idx, 2]

        df_batch = pd.DataFrame({
            "norad_id": norad_col,
            "timestep": ts_col,
            "x_km":     x_col,
            "y_km":     y_col,
            "z_km":     z_col,
        })
        df_batch.to_csv(OUTPUT_CSV, mode="a", index=False, header=False,
                        float_format="%.4f")

        n_written  += len(df_batch)
        n_objects  += b_valid

        if n_objects % PRINT_EVERY < batch_size or batch_end == N_cat:
            elapsed = time.perf_counter() - t_start
            rate    = n_objects / elapsed if elapsed > 0 else 0
            eta_s   = (N_cat - n_objects) / rate if rate > 0 else 0
            pct     = 100 * n_objects / N_cat
            print(f"  [{n_objects:>6,}/{N_cat:,}]  {pct:5.1f}%  "
                  f"rows_written={n_written:,}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_s:.0f}s")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - t_wall
    file_mb       = OUTPUT_CSV.stat().st_size / (1024 ** 2)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Catalog objects   : {N_cat:,}")
    print(f"  SGP4 error steps  : {n_errors:,}")
    print(f"  Rows written      : {n_written:,}")
    print(f"  File size         : {file_mb:.0f} MB")
    print(f"  Wall time         : {total_elapsed:.0f} s  "
          f"({total_elapsed/60:.1f} min)")
    print(f"  Output            : {OUTPUT_CSV}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
