"""
propagate_orbits.py
-------------------
Propagates TLE orbits using SGP4 and stores GCRF cartesian state vectors.

Two operating modes selected by the --synthetic flag:

  Default mode (real TLEs, Owens-Fahrner 2025 pipeline)
  -------------------------------------------------------
  data/propagated_candidates.csv
      Full 3-day trajectory for Shell-3 real candidates.
      Columns: norad_id, epoch_utc, x_km, y_km, z_km,
               vx_kms, vy_kms, vz_kms, altitude_km, error
      Rows: N_candidates x 4321 timesteps (60 s step).

  data/propagated_catalog.csv
      One-row snapshot for every catalog object at simulation start.
      Columns: norad_id, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms,
               altitude_km, error

  Synthetic mode  (--synthetic, Arnas 2021 Shell 3 full constellation)
  ----------------------------------------------------------------------
  Reads:   data/shell3_synthetic.tle  (1,656 satellites, NORAD 90001-91656)
  Writes:  data/propagated_candidates.csv   <-- OVERWRITES the default file
  Columns: norad_id, timestep, x_km, y_km, z_km
  Rows:    1,656 x 4,321 = 7,152,696

  Uses SatrecArray for fully vectorised propagation (N x T in one NumPy
  call per batch) and tqdm for a live progress bar.

Simulation parameters (Owens-Fahrner 2025, Section 4):
    Duration  : 3 days (259,200 s)
    Time step : 60 s
    Frame     : GCRF (approx ECI J2000 for SGP4 outputs)

Usage:
    python src/propagate_orbits.py
    python src/propagate_orbits.py --days 3 --step 60
    python src/propagate_orbits.py --synthetic
    python src/propagate_orbits.py --synthetic --batch-size 300
"""

import argparse
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
CANDIDATE_TLE_PATH = DATA_DIR / "shell_550km.tle"
CATALOG_TLE_PATH   = DATA_DIR / "leo_catalog.tle"
SYNTHETIC_TLE_PATH = DATA_DIR / "shell3_synthetic.tle"   # Arnas 2021 constellation
OUTPUT_CANDIDATES  = DATA_DIR / "propagated_candidates.csv"
OUTPUT_CATALOG     = DATA_DIR / "propagated_catalog.csv"
OUTPUT_PLOT        = Path(__file__).parent.parent / "results" / "ground_tracks.png"

# Simulation defaults (Owens-Fahrner 2025, Section 4)
DEFAULT_DAYS  = 3
DEFAULT_STEP  = 60.0   # seconds

# Batch size for SatrecArray vectorised propagation (--synthetic mode).
# 200 satellites x 4321 timesteps x 3 coords x 8 bytes ~ 20 MB per batch.
DEFAULT_BATCH_SIZE = 200

# Earth constants
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# TLE loading
# ---------------------------------------------------------------------------

def load_tles(tle_path: Path) -> list[dict]:
    """
    Parse a 2-line TLE file and return a list of satellite records.

    Parameters
    ----------
    tle_path : Path

    Returns
    -------
    list of dict, each with 'norad_id', 'line1', 'line2'
    """
    lines = [ln.strip() for ln in tle_path.read_text().splitlines() if ln.strip()]
    satellites = []
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if not (l1.startswith('1 ') and l2.startswith('2 ')):
            continue
        satellites.append({
            'norad_id': l1[2:7].strip(),
            'line1': l1,
            'line2': l2,
        })
    return satellites


# ---------------------------------------------------------------------------
# SGP4 propagation helpers
# ---------------------------------------------------------------------------

def build_time_arrays(
    start_time: datetime,
    duration_seconds: float,
    step_seconds: float,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """
    Build datetime and Julian-date arrays for sgp4_array calls.

    Returns
    -------
    datetimes : list of datetime (UTC, timezone-aware)
    jd_array  : ndarray, shape (T,) — integer Julian date parts
    fr_array  : ndarray, shape (T,) — fractional Julian date parts
    """
    n_steps = int(duration_seconds / step_seconds) + 1
    datetimes, jd_list, fr_list = [], [], []
    for i in range(n_steps):
        t = start_time + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        datetimes.append(t)
        jd_list.append(jd)
        fr_list.append(fr)
    return datetimes, np.array(jd_list), np.array(fr_list)


def propagate_satellite_full(
    sat_record: dict,
    datetimes: list[datetime],
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> list[dict]:
    """
    Propagate one satellite at all T timesteps via sgp4_array (vectorised).

    Returns one dict per timestep with full state vector in GCRF (km, km/s).
    """
    satellite = Satrec.twoline2rv(sat_record['line1'], sat_record['line2'])
    norad_id = sat_record['norad_id']

    errors, r_all, v_all = satellite.sgp4_array(jd_array, fr_array)
    r_all = np.array(r_all)   # (T, 3)
    v_all = np.array(v_all)   # (T, 3)
    errors = np.array(errors)

    rows = []
    for i, (t, err) in enumerate(zip(datetimes, errors)):
        if err != 0:
            rows.append({
                'norad_id': norad_id,
                'epoch_utc': t.isoformat(),
                'x_km': np.nan, 'y_km': np.nan, 'z_km': np.nan,
                'vx_kms': np.nan, 'vy_kms': np.nan, 'vz_kms': np.nan,
                'altitude_km': np.nan,
                'error': int(err),
            })
        else:
            r = r_all[i]
            v = v_all[i]
            altitude_km = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - EARTH_RADIUS_KM
            rows.append({
                'norad_id': norad_id,
                'epoch_utc': t.isoformat(),
                'x_km': r[0], 'y_km': r[1], 'z_km': r[2],
                'vx_kms': v[0], 'vy_kms': v[1], 'vz_kms': v[2],
                'altitude_km': altitude_km,
                'error': 0,
            })
    return rows


def propagate_satellite_snapshot(
    sat_record: dict,
    jd0: float,
    fr0: float,
) -> dict:
    """
    Propagate one satellite to a single epoch (snapshot).

    Used to build the propagated_catalog.csv snapshot without storing
    the full 3-day trajectory for every catalog object.

    Returns one dict with the state at the given epoch, or error fields
    if SGP4 fails.
    """
    satellite = Satrec.twoline2rv(sat_record['line1'], sat_record['line2'])
    norad_id = sat_record['norad_id']
    err, r, v = satellite.sgp4(jd0, fr0)

    if err != 0:
        return {
            'norad_id': norad_id,
            'x_km': np.nan, 'y_km': np.nan, 'z_km': np.nan,
            'vx_kms': np.nan, 'vy_kms': np.nan, 'vz_kms': np.nan,
            'altitude_km': np.nan,
            'error': err,
        }

    altitude_km = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - EARTH_RADIUS_KM
    return {
        'norad_id': norad_id,
        'x_km': r[0], 'y_km': r[1], 'z_km': r[2],
        'vx_kms': v[0], 'vy_kms': v[1], 'vz_kms': v[2],
        'altitude_km': altitude_km,
        'error': 0,
    }


# ---------------------------------------------------------------------------
# Ground-track visualisation
# ---------------------------------------------------------------------------

def eci_to_lat_lon_approx(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Approximate ECI → latitude/longitude (no Earth-rotation correction).
    Suitable for ground-track shape visualisation only.
    """
    r = math.sqrt(x**2 + y**2 + z**2)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def plot_ground_tracks(df: pd.DataFrame, output_path: Path) -> None:
    """Plot ground tracks for all candidate satellites."""
    valid = df[df['error'] == 0].copy()
    lats, lons = [], []
    for _, row in valid.iterrows():
        la, lo = eci_to_lat_lon_approx(row['x_km'], row['y_km'], row['z_km'])
        lats.append(la)
        lons.append(lo)
    valid = valid.copy()
    valid['lat'] = lats
    valid['lon'] = lons

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')

    norad_ids = valid['norad_id'].unique()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(norad_ids)))
    for norad_id, color in zip(norad_ids, colors):
        sub = valid[valid['norad_id'] == norad_id]
        ax.scatter(sub['lon'], sub['lat'], s=0.3, color=color, alpha=0.5)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (deg)', color='white')
    ax.set_ylabel('Latitude (deg)', color='white')
    ax.set_title(
        f'Approximate Ground Tracks — Shell 3 ({len(norad_ids)} candidates, 3 days)',
        color='white',
    )
    ax.tick_params(colors='white')
    ax.spines[:].set_color('white')
    ax.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    plt.close()
    print(f"  Ground track plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_candidate_summary(df: pd.DataFrame) -> None:
    valid = df[df['error'] == 0]
    print()
    print("=" * 60)
    print("CANDIDATE PROPAGATION SUMMARY")
    print("=" * 60)
    print(f"  Total state vectors : {len(df):,}")
    print(f"  Valid               : {len(valid):,}")
    print(f"  Errors / decayed    : {len(df) - len(valid):,}")
    print()
    print(f"  {'NORAD':>8}  {'Mean alt':>10}  {'Min alt':>8}  {'Max alt':>8}  {'Steps':>7}")
    for norad_id, grp in valid.groupby('norad_id'):
        print(f"  {norad_id:>8}  "
              f"{grp['altitude_km'].mean():>10.1f}  "
              f"{grp['altitude_km'].min():>8.1f}  "
              f"{grp['altitude_km'].max():>8.1f}  "
              f"{len(grp):>7,}")
    print("=" * 60)


def print_catalog_summary(df: pd.DataFrame) -> None:
    valid = df[df['error'] == 0]
    print()
    print("=" * 60)
    print("CATALOG SNAPSHOT SUMMARY")
    print("=" * 60)
    print(f"  Total objects       : {len(df):,}")
    print(f"  Valid               : {len(valid):,}")
    print(f"  Errors / decayed    : {len(df) - len(valid):,}")
    alts = valid['altitude_km']
    print(f"  Altitude range      : {alts.min():.0f} – {alts.max():.0f} km")
    print(f"  Below 600 km        : {(alts < 600).sum():,}")
    print(f"  600–1200 km         : {((alts >= 600) & (alts < 1200)).sum():,}")
    print(f"  1200–2000 km        : {((alts >= 1200) & (alts <= 2000)).sum():,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Vectorised synthetic propagation  (--synthetic mode)
# ---------------------------------------------------------------------------

def propagate_synthetic_vectorized(
    tle_path: Path,
    output_path: Path,
    duration_days: float,
    step_seconds: float,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, float, float, float]:
    """
    Propagate a large synthetic TLE constellation using SatrecArray.

    Strategy
    --------
    Satellites are processed in batches of `batch_size`.  For each batch,
    SatrecArray.sgp4(jd_array, fr_array) propagates all batch satellites
    simultaneously in a single NumPy call, returning arrays of shape
    (batch_n, T, 3).  The results are reshaped into a DataFrame and
    streamed to the output CSV (append mode) to avoid holding the full
    7 M-row dataset in memory at once.

    Parameters
    ----------
    tle_path : Path
        Input TLE file (e.g. shell3_synthetic.tle).
    output_path : Path
        Destination CSV (norad_id, timestep, x_km, y_km, z_km).
    duration_days : float
        Simulation duration in days.
    step_seconds : float
        Propagation timestep in seconds.
    batch_size : int
        Number of satellites per SatrecArray call.

    Returns
    -------
    n_errors : int
        Total SGP4 error count across all satellites and timesteps.
    alt_min : float
        Minimum altitude (km) over all valid positions.
    alt_max : float
        Maximum altitude (km) over all valid positions.
    elapsed : float
        Wall-clock propagation time in seconds.
    """
    # --- Load TLEs --------------------------------------------------------
    sat_records = load_tles(tle_path)
    N = len(sat_records)

    norad_ids = np.array([int(s['norad_id']) for s in sat_records], dtype=np.int32)
    satrecs   = [Satrec.twoline2rv(s['line1'], s['line2']) for s in sat_records]

    # --- Build time arrays anchored to the TLE epoch ----------------------
    # Using datetime.now() would create a mismatch between timestep indices
    # stored in propagated_candidates.csv and the jd_array built in
    # compute_pc.py from the TLE epoch.  Anchor t=0 to the TLE epoch instead.
    duration_s = duration_days * 86400.0
    T          = int(duration_s / step_seconds) + 1   # 4321 for 3d/60s

    l1_ref     = sat_records[0]['line1']
    y2_ref     = int(l1_ref[18:20])
    doy_ref    = float(l1_ref[20:32])
    year_ref   = (2000 + y2_ref) if y2_ref < 57 else (1900 + y2_ref)
    start_time = (datetime(year_ref, 1, 1, tzinfo=timezone.utc)
                  + timedelta(days=doy_ref - 1.0))

    _, jd_array, fr_array = build_time_arrays(start_time, duration_s, step_seconds)
    jd_array = jd_array.astype(np.float64)
    fr_array = fr_array.astype(np.float64)

    # --- Streaming CSV write ----------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('norad_id,timestep,x_km,y_km,z_km\n')

    # Timestep index column, shared across batches
    ts_tile = np.tile(np.arange(T, dtype=np.int32), batch_size)  # pre-alloc max

    n_errors = 0
    alt_min  =  math.inf
    alt_max  = -math.inf
    t0 = time.perf_counter()

    n_batches = math.ceil(N / batch_size)
    pbar = tqdm(
        range(0, N, batch_size),
        total=n_batches,
        desc='Propagating',
        unit='batch',
        ncols=72,
    )

    for batch_start in pbar:
        batch_end    = min(batch_start + batch_size, N)
        b_n          = batch_end - batch_start          # satellites this batch
        b_satrecs    = satrecs[batch_start:batch_end]
        b_norads     = norad_ids[batch_start:batch_end] # shape (b_n,)

        # -------------------------------------------------------------------
        # SatrecArray.sgp4 — fully vectorised:
        #   e  shape (b_n, T)   error codes (0 = success)
        #   r  shape (b_n, T, 3) GCRF positions in km
        #   v  shape (b_n, T, 3) GCRF velocities in km/s  (unused here)
        # -------------------------------------------------------------------
        sat_arr = SatrecArray(b_satrecs)
        e_batch, r_batch, _ = sat_arr.sgp4(jd_array, fr_array)

        r_np = np.asarray(r_batch, dtype=np.float64)   # (b_n, T, 3)
        e_np = np.asarray(e_batch, dtype=np.int32)     # (b_n, T)

        # Accumulate error count
        n_errors += int((e_np != 0).sum())

        # Altitude stats (vectorised, valid rows only)
        if e_np.size > 0:
            r_flat = r_np.reshape(-1, 3)               # (b_n*T, 3)
            e_flat = e_np.ravel()                      # (b_n*T,)
            valid  = e_flat == 0
            if valid.any():
                alts = np.linalg.norm(r_flat[valid], axis=1) - EARTH_RADIUS_KM
                if alts.min() < alt_min:
                    alt_min = float(alts.min())
                if alts.max() > alt_max:
                    alt_max = float(alts.max())

        # -------------------------------------------------------------------
        # Build output columns — no Python loops
        #   norad column: each norad ID repeated T times
        #   ts column:    timestep indices 0..T-1 tiled b_n times
        # -------------------------------------------------------------------
        norad_col = np.repeat(b_norads, T)             # (b_n*T,)
        ts_col    = ts_tile[:b_n * T]                  # slice pre-allocated tile
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

        # Append to CSV (no header — already written above)
        df_batch.to_csv(output_path, mode='a', index=False, header=False,
                        float_format='%.4f')

        pbar.set_postfix({'sats': f'{batch_end}/{N}', 'errs': n_errors})

    elapsed = time.perf_counter() - t0
    return n_errors, alt_min, alt_max, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGP4 orbit propagator — Owens-Fahrner 2025 / Arnas 2021"
    )
    parser.add_argument(
        '--days', type=float, default=DEFAULT_DAYS,
        help=f'Simulation duration in days (default: {DEFAULT_DAYS})'
    )
    parser.add_argument(
        '--step', type=float, default=DEFAULT_STEP,
        help=f'Time step in seconds (default: {DEFAULT_STEP:.0f})'
    )
    parser.add_argument(
        '--no-catalog', action='store_true',
        help='Skip catalog snapshot propagation (real-TLE mode only)'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help=(
            'Propagate the Arnas 2021 synthetic Shell 3 constellation '
            '(shell3_synthetic.tle, 1656 sats) using SatrecArray vectorisation. '
            'Writes norad_id,timestep,x_km,y_km,z_km to propagated_candidates.csv '
            '(OVERWRITES the real-TLE output file).'
        )
    )
    parser.add_argument(
        '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help=f'Satellites per SatrecArray call in --synthetic mode '
             f'(default: {DEFAULT_BATCH_SIZE})'
    )
    args = parser.parse_args()

    duration_s = args.days * 86400.0
    n_steps    = int(duration_s / args.step) + 1

    # ======================================================================
    # SYNTHETIC MODE
    # ======================================================================
    if args.synthetic:
        print("=" * 68)
        print("SGP4 Orbit Propagator — Arnas 2021 Synthetic Shell 3")
        print(f"  Input      : {SYNTHETIC_TLE_PATH.name}")
        print(f"  Duration   : {args.days} days  ({duration_s / 3600:.0f} h)")
        print(f"  Timestep   : {args.step:.0f} s   ->  {n_steps:,} steps per satellite")
        print(f"  Batch size : {args.batch_size} satellites per SatrecArray call")
        print(f"  Output     : {OUTPUT_CANDIDATES.name}  (overwrites real-TLE file)")
        print("=" * 68)

        if not SYNTHETIC_TLE_PATH.exists():
            print(f"\n  ERROR: {SYNTHETIC_TLE_PATH} not found.")
            print("  Run 'python src/generate_candidates.py' first.")
            return

        # Count satellites before running
        sat_records = load_tles(SYNTHETIC_TLE_PATH)
        N = len(sat_records)
        total_rows = N * n_steps
        print(f"\n  Loaded         : {N:,} satellites  "
              f"(NORAD {int(sat_records[0]['norad_id'])} .. "
              f"{int(sat_records[-1]['norad_id'])})")
        print(f"  Total rows     : {total_rows:,}  ({N:,} x {n_steps:,})")
        print()

        n_errors, alt_min, alt_max, elapsed = propagate_synthetic_vectorized(
            SYNTHETIC_TLE_PATH,
            OUTPUT_CANDIDATES,
            args.days,
            args.step,
            batch_size=args.batch_size,
        )

        # ---------------------------------------------------------------
        # Sanity checks
        # ---------------------------------------------------------------
        print()
        print("=" * 68)
        print("SANITY CHECKS")
        print("=" * 68)
        print(f"  SGP4 errors        : {n_errors}"
              f"  ({'PASS - expect 0' if n_errors == 0 else 'FAIL - unexpected errors'})")
        print(f"  Altitude range     : {alt_min:.1f} - {alt_max:.1f} km"
              f"  ({'PASS' if 540 <= alt_min and alt_max <= 560 else 'WARN - outside 540-560 km'})"
              f"  (expected ~548-552 km, nearly circular)")
        print(f"  Computation time   : {elapsed:.1f} s")

        file_size_mb = OUTPUT_CANDIDATES.stat().st_size / (1024 ** 2)
        print(f"  Output file        : {OUTPUT_CANDIDATES}")
        print(f"  File size          : {file_size_mb:.1f} MB  ({total_rows:,} rows)")
        print("=" * 68)
        print("Done.")
        return

    # ======================================================================
    # DEFAULT MODE  (real TLEs, Owens-Fahrner 2025)
    # ======================================================================
    print("=" * 65)
    print("SGP4 Orbit Propagator — Owens-Fahrner 2025")
    print(f"  Duration  : {args.days} days  ({duration_s / 3600:.0f} h)")
    print(f"  Time step : {args.step:.0f} s")
    print(f"  Timesteps : {n_steps:,} per object")
    print("=" * 65)

    start_time = datetime.now(timezone.utc)
    print(f"\n  Simulation start (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    datetimes, jd_array, fr_array = build_time_arrays(start_time, duration_s, args.step)
    jd0, fr0 = jd_array[0], fr_array[0]

    # ------------------------------------------------------------------
    # 1. Propagate candidates — full 3-day trajectory
    # ------------------------------------------------------------------
    print(f"\n--- Propagating candidates from {CANDIDATE_TLE_PATH.name} ---")
    candidates = load_tles(CANDIDATE_TLE_PATH)
    print(f"  Loaded {len(candidates)} candidate TLEs")
    print(f"  Total state vectors: {len(candidates) * n_steps:,}")

    candidate_rows = []
    for i, sat in enumerate(candidates):
        rows = propagate_satellite_full(sat, datetimes, jd_array, fr_array)
        valid_n = sum(1 for r in rows if r['error'] == 0)
        print(f"  [{i+1:02d}/{len(candidates)}] NORAD {sat['norad_id']}  "
              f"valid={valid_n}/{n_steps}")
        candidate_rows.extend(rows)

    df_cand = pd.DataFrame(candidate_rows)
    OUTPUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    df_cand.to_csv(OUTPUT_CANDIDATES, index=False)
    print(f"\n  Saved {len(df_cand):,} rows to: {OUTPUT_CANDIDATES}")
    print_candidate_summary(df_cand)

    # ------------------------------------------------------------------
    # 2. Propagate catalog — snapshot at simulation start epoch
    # ------------------------------------------------------------------
    if not args.no_catalog:
        print(f"\n--- Propagating catalog snapshot from {CATALOG_TLE_PATH.name} ---")
        catalog = load_tles(CATALOG_TLE_PATH)
        print(f"  Loaded {len(catalog):,} catalog TLEs")
        print(f"  Propagating to epoch {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC...")

        catalog_rows = []
        for sat in catalog:
            catalog_rows.append(propagate_satellite_snapshot(sat, jd0, fr0))

        df_cat = pd.DataFrame(catalog_rows)
        OUTPUT_CATALOG.parent.mkdir(parents=True, exist_ok=True)
        df_cat.to_csv(OUTPUT_CATALOG, index=False)
        print(f"  Saved {len(df_cat):,} rows to: {OUTPUT_CATALOG}")
        print_catalog_summary(df_cat)

    # ------------------------------------------------------------------
    # 3. Ground-track visualisation for candidates
    # ------------------------------------------------------------------
    print("\nGenerating ground track plot...")
    plot_ground_tracks(df_cand, OUTPUT_PLOT)

    print("\nDone.")


if __name__ == "__main__":
    main()
