"""
propagate_orbits.py
-------------------
Propagates TLE orbits using SGP4 and stores GCRF cartesian state vectors.

Produces two output files:

  data/propagated_candidates.csv
      Full 3-day trajectory for each of the 20 Shell-3 candidate satellites.
      Columns: norad_id, epoch_utc, x_km, y_km, z_km,
               vx_kms, vy_kms, vz_kms, altitude_km, error
      Rows: 20 candidates × 4321 timesteps (60 s step) = 86,420 rows.

  data/propagated_catalog.csv
      One-row snapshot for every catalog object, propagated to the
      simulation start epoch.  Used for altitude verification and
      spatial distribution analysis.  The full catalog trajectory is
      NOT stored (22,000 × 4,321 rows ≈ 95 M rows); compute_pc.py
      propagates catalog objects on-the-fly via sgp4_array.
      Columns: norad_id, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms,
               altitude_km, error

Simulation parameters (Owens-Fahrner 2025, Section 4):
    Duration  : 3 days (259,200 s)
    Time step : 60 s
    Frame     : GCRF (≈ ECI J2000 for SGP4 outputs)

Usage:
    python src/propagate_orbits.py
    python src/propagate_orbits.py --days 3 --step 60
"""

import argparse
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sgp4.api import Satrec, jday

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
CANDIDATE_TLE_PATH = DATA_DIR / "shell_550km.tle"
CATALOG_TLE_PATH   = DATA_DIR / "leo_catalog.tle"
OUTPUT_CANDIDATES  = DATA_DIR / "propagated_candidates.csv"
OUTPUT_CATALOG     = DATA_DIR / "propagated_catalog.csv"
OUTPUT_PLOT        = Path(__file__).parent.parent / "results" / "ground_tracks.png"

# Simulation defaults (Owens-Fahrner 2025, Section 4)
DEFAULT_DAYS  = 3
DEFAULT_STEP  = 60.0   # seconds

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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGP4 orbit propagator — Owens-Fahrner 2025 pipeline"
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
        help='Skip catalog snapshot propagation (faster for candidate-only runs)'
    )
    args = parser.parse_args()

    duration_s = args.days * 86400.0
    n_steps = int(duration_s / args.step) + 1

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
