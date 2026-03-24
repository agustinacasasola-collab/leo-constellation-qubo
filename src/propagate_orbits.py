"""
propagate_orbits.py
-------------------
Propagates TLE orbits using SGP4 and saves ECI state vectors.

For each object in data/shell_550km.tle, computes position (km) and
velocity (km/s) in the Earth-Centered Inertial (ECI) frame at regular
time steps over a configurable simulation window.

Output: data/propagated_states.csv
    norad_id, epoch_utc, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms, altitude_km

Also saves a ground-track plot to results/ground_tracks.png.

Usage:
    python src/propagate_orbits.py
    python src/propagate_orbits.py --hours 12 --step 30
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
TLE_PATH = Path(__file__).parent.parent / "data" / "shell_550km.tle"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "propagated_states.csv"
OUTPUT_PLOT = Path(__file__).parent.parent / "results" / "ground_tracks.png"
OUTPUT_PLOT_FILTERED = Path(__file__).parent.parent / "results" / "ground_tracks_filtered.png"

# Altitude filter bounds for the 550 km shell
ALT_FILTER_MIN_KM = 540.0
ALT_FILTER_MAX_KM = 560.0

# Earth constants
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# TLE parsing
# ---------------------------------------------------------------------------

def load_tles(tle_path: Path) -> list[dict]:
    """
    Parse a TLE file and return a list of satellite records.

    Each TLE object occupies exactly 2 lines (no name line in this format).
    Lines starting with '1 ' are Line 1, lines starting with '2 ' are Line 2.

    Parameters
    ----------
    tle_path : Path
        Path to the .tle file.

    Returns
    -------
    list of dict
        Each dict has keys: 'norad_id' (str), 'line1' (str), 'line2' (str).
    """
    lines = [l.strip() for l in tle_path.read_text().splitlines() if l.strip()]

    satellites = []
    for i in range(0, len(lines) - 1, 2):
        line1 = lines[i]
        line2 = lines[i + 1]
        if not (line1.startswith('1 ') and line2.startswith('2 ')):
            continue
        norad_id = line1[2:7].strip()
        satellites.append({'norad_id': norad_id, 'line1': line1, 'line2': line2})

    return satellites


# ---------------------------------------------------------------------------
# SGP4 propagation
# ---------------------------------------------------------------------------

def propagate_satellite(
    sat_record: dict,
    start_time: datetime,
    duration_hours: float,
    step_seconds: float,
) -> list[dict]:
    """
    Propagate one satellite over the simulation window using SGP4.

    SGP4 is the standard analytical propagator for LEO objects. It models
    the effects of Earth's oblateness (J2, J3, J4 zonal harmonics), atmospheric
    drag, and solar/lunar perturbations. Accuracy degrades for TLEs older than
    a few days; this is why we filter for EPOCH > now-30 in fetch_tles.py.

    The output is in the ECI (Earth-Centered Inertial) frame:
    - Origin: Earth's centre of mass
    - X-axis: vernal equinox direction (fixed to stars, not rotating with Earth)
    - Z-axis: Earth's north pole
    - Y-axis: completes right-hand system

    Parameters
    ----------
    sat_record : dict
        Dict with keys 'norad_id', 'line1', 'line2'.
    start_time : datetime
        UTC start time for propagation (timezone-aware).
    duration_hours : float
        Total propagation window in hours.
    step_seconds : float
        Time step between state vector samples (seconds).

    Returns
    -------
    list of dict
        One entry per time step, with keys:
        norad_id, epoch_utc, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms,
        altitude_km, error (0 = success).
    """
    satellite = Satrec.twoline2rv(sat_record['line1'], sat_record['line2'])
    norad_id = sat_record['norad_id']

    num_steps = int(duration_hours * 3600 / step_seconds) + 1
    results = []

    for i in range(num_steps):
        t = start_time + timedelta(seconds=i * step_seconds)

        # Convert datetime to Julian Date (integer + fractional parts).
        # SGP4 requires this split to preserve floating-point precision.
        jd, fr = jday(
            t.year, t.month, t.day,
            t.hour, t.minute, t.second + t.microsecond / 1e6
        )

        # Propagate: returns error code, position (km), velocity (km/s).
        # Error codes: 0 = OK, 1 = mean eccentricity out of range,
        # 2 = mean motion < 0, 3 = pert elements < 0, 4 = semi-latus < 0,
        # 5 = orbit decayed.
        error, r, v = satellite.sgp4(jd, fr)

        if error != 0:
            # Skip decayed or invalid states but record the error.
            results.append({
                'norad_id': norad_id,
                'epoch_utc': t.isoformat(),
                'x_km': np.nan, 'y_km': np.nan, 'z_km': np.nan,
                'vx_kms': np.nan, 'vy_kms': np.nan, 'vz_kms': np.nan,
                'altitude_km': np.nan,
                'error': error,
            })
            continue

        # Altitude above Earth's surface (assumes spherical Earth).
        altitude_km = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - EARTH_RADIUS_KM

        results.append({
            'norad_id': norad_id,
            'epoch_utc': t.isoformat(),
            'x_km': r[0],
            'y_km': r[1],
            'z_km': r[2],
            'vx_kms': v[0],
            'vy_kms': v[1],
            'vz_kms': v[2],
            'altitude_km': altitude_km,
            'error': 0,
        })

    return results


def eci_to_geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """
    Convert ECI position to geographic latitude and longitude.

    Uses a simplified spherical Earth model (sufficient for ground-track
    plotting; production code should use WGS-84 oblate spheroid).

    This conversion requires knowing the Greenwich Sidereal Time (GST) to
    rotate from ECI to ECEF (Earth-Centered, Earth-Fixed). Here we use a
    simple approximation for visualisation purposes.

    Parameters
    ----------
    x, y, z : float
        ECI position components (km).

    Returns
    -------
    tuple of (latitude_deg, longitude_deg)
    """
    # Geocentric latitude (spherical approximation)
    r = math.sqrt(x**2 + y**2 + z**2)
    lat = math.degrees(math.asin(z / r))

    # Longitude in ECI frame (not accounting for Earth rotation — approximate)
    lon = math.degrees(math.atan2(y, x))

    return lat, lon


# ---------------------------------------------------------------------------
# Altitude filter
# ---------------------------------------------------------------------------

def filter_by_mean_altitude(
    df: pd.DataFrame,
    alt_min_km: float = ALT_FILTER_MIN_KM,
    alt_max_km: float = ALT_FILTER_MAX_KM,
    max_count: int = 20,
    center_km: float = 550.0,
) -> pd.DataFrame:
    """
    Filter propagated states to keep satellites whose mean altitude falls
    within [alt_min_km, alt_max_km], then select the ``max_count`` objects
    whose mean altitude is closest to ``center_km``.

    Selecting by proximity to the shell centre (550 km) is physically
    motivated: it prioritises the objects most representative of the
    congested Starlink shell, discarding those at the fringe of the band.

    Mean altitude is computed from valid (error=0) time steps only.

    Parameters
    ----------
    df : pd.DataFrame
        Full propagated states DataFrame from propagate_all().
    alt_min_km : float
        Lower bound for mean altitude (km). Default 540.0.
    alt_max_km : float
        Upper bound for mean altitude (km). Default 560.0.
    max_count : int
        Maximum number of satellites to keep after filtering.
        Selects the ``max_count`` objects nearest to ``center_km``.
        Default 20.
    center_km : float
        Reference altitude for proximity ranking (km). Default 550.0.

    Returns
    -------
    pd.DataFrame
        Subset of df containing only rows belonging to the selected
        satellites, ordered by proximity to center_km.
    """
    valid = df[df['error'] == 0]
    mean_alts = valid.groupby('norad_id')['altitude_km'].mean()

    # Step 1: band filter
    in_band = mean_alts[
        (mean_alts >= alt_min_km) & (mean_alts <= alt_max_km)
    ]

    # Step 2: rank by distance to shell centre, keep best max_count
    ranked = in_band.reindex(
        (in_band - center_km).abs().sort_values().index
    )
    selected = ranked.iloc[:max_count]

    print()
    print("=" * 60)
    print("ALTITUDE FILTER  (mean altitude per satellite)")
    print(f"  Band  : {alt_min_km} – {alt_max_km} km")
    print(f"  Centre: {center_km} km  |  Keep closest: {max_count}")
    print("=" * 60)
    print(f"  {'NORAD ID':<12} {'Mean alt (km)':>15}  {'dist 550km':>10}  {'Status':>8}")
    print(f"  {'-'*12} {'-'*15}  {'-'*10}  {'-'*8}")
    for norad_id, mean_alt in mean_alts.sort_values().items():
        delta = abs(mean_alt - center_km)
        if norad_id not in in_band.index:
            status = "out-band"
        elif norad_id in selected.index:
            status = "SELECTED"
        else:
            status = "trimmed"
        print(f"  {norad_id:<12} {mean_alt:>15.2f}  {delta:>10.2f}  {status:>8}")

    print()
    print(f"  In band : {len(in_band)} / {len(mean_alts)} satellites")
    print(f"  Selected: {len(selected)} (closest to {center_km} km)")
    print(f"  NORAD IDs: {sorted(selected.index.tolist())}")
    print("=" * 60)

    return df[df['norad_id'].isin(selected.index)].copy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ground_tracks(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Plot approximate ground tracks for propagated objects.

    Note: this uses ECI longitude directly (no Earth rotation correction),
    so tracks are approximate — suitable for visualising orbital coverage
    patterns but not precise geographic positions.

    Parameters
    ----------
    df : pd.DataFrame
        Propagated states (full or filtered) from propagate_all().
    output_path : Path
        Path to save the PNG figure.
    title_suffix : str, optional
        Text appended to the plot title (e.g. " — filtered 540-560 km").
    """
    valid = df[df['error'] == 0].copy()

    # Compute geodetic coordinates
    lats, lons = [], []
    for _, row in valid.iterrows():
        lat, lon = eci_to_geodetic(row['x_km'], row['y_km'], row['z_km'])
        lats.append(lat)
        lons.append(lon)
    valid = valid.copy()
    valid['lat'] = lats
    valid['lon'] = lons

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Ground tracks ---
    ax = axes[0]
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')

    norad_ids = valid['norad_id'].unique()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(norad_ids)))

    for norad_id, color in zip(norad_ids, colors):
        subset = valid[valid['norad_id'] == norad_id]
        ax.scatter(subset['lon'], subset['lat'], s=0.5, color=color, alpha=0.6)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (°)', color='white', fontsize=10)
    ax.set_ylabel('Latitude (°)', color='white', fontsize=10)
    ax.set_title(
        f'Approximate Ground Tracks — Shell 3 (~550 km)  |  '
        f'{len(norad_ids)} objects{title_suffix}',
        color='white', fontsize=12
    )
    ax.tick_params(colors='white')
    ax.spines[:].set_color('white')
    ax.grid(True, alpha=0.2, color='white')
    ax.axhline(0, color='white', alpha=0.3, linewidth=0.5)

    # --- Altitude over time (first 6 objects) ---
    ax2 = axes[1]
    ax2.set_facecolor('#0a0a1a')

    for norad_id, color in zip(norad_ids[:6], colors[:6]):
        subset = valid[valid['norad_id'] == norad_id].copy()
        subset = subset.reset_index(drop=True)
        times_h = [i * (subset.index[1] if len(subset) > 1 else 1) for i in range(len(subset))]
        # Use row index as time proxy (each row = one time step)
        ax2.plot(subset.index, subset['altitude_km'],
                 color=color, linewidth=1.0, label=f'NORAD {norad_id}')

    ax2.set_xlabel('Time step', color='white', fontsize=10)
    ax2.set_ylabel('Altitude (km)', color='white', fontsize=10)
    ax2.set_title('Altitude vs Time (first 6 objects)', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
               loc='upper right')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a')
    plt.close()
    print(f"  Ground track plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def propagate_all(
    tle_path: Path,
    duration_hours: float,
    step_seconds: float,
) -> pd.DataFrame:
    """
    Propagate all TLE objects and return a combined DataFrame.

    Parameters
    ----------
    tle_path : Path
        Path to the .tle file.
    duration_hours : float
        Propagation window in hours.
    step_seconds : float
        Time step in seconds.

    Returns
    -------
    pd.DataFrame
        Combined state vectors for all objects.
    """
    satellites = load_tles(tle_path)
    print(f"  Loaded {len(satellites)} TLE objects from {tle_path.name}")

    start_time = datetime.now(timezone.utc)
    print(f"  Start time (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {duration_hours} hours  |  Step: {step_seconds} s")
    steps_per_sat = int(duration_hours * 3600 / step_seconds) + 1
    print(f"  Time steps per object: {steps_per_sat}")
    print(f"  Total state vectors: {len(satellites) * steps_per_sat:,}")
    print()

    all_rows = []
    for i, sat in enumerate(satellites):
        rows = propagate_satellite(sat, start_time, duration_hours, step_seconds)
        valid = sum(1 for r in rows if r['error'] == 0)
        errors = len(rows) - valid
        status = f"ok={valid}" + (f" errors={errors}" if errors else "")
        print(f"  [{i+1:02d}/{len(satellites)}] NORAD {sat['norad_id']}  {status}")
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def print_summary(df: pd.DataFrame) -> None:
    """Print a statistical summary of the propagated states."""
    valid = df[df['error'] == 0]
    print()
    print("=" * 60)
    print("PROPAGATION SUMMARY")
    print("=" * 60)
    print(f"  Total state vectors : {len(df):,}")
    print(f"  Valid               : {len(valid):,}")
    print(f"  Errors / decayed    : {len(df) - len(valid):,}")
    print()
    print("  Altitude statistics (km):")
    print(f"    Min  : {valid['altitude_km'].min():.2f}")
    print(f"    Max  : {valid['altitude_km'].max():.2f}")
    print(f"    Mean : {valid['altitude_km'].mean():.2f}")
    print(f"    Std  : {valid['altitude_km'].std():.2f}")
    print()
    print("  Per-object altitude range:")
    for norad_id, group in valid.groupby('norad_id'):
        print(f"    NORAD {norad_id:>6}  "
              f"alt = [{group['altitude_km'].min():.1f}, "
              f"{group['altitude_km'].max():.1f}] km  "
              f"mean = {group['altitude_km'].mean():.1f} km")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate TLE orbits with SGP4"
    )
    parser.add_argument(
        '--hours', type=float, default=24.0,
        help='Propagation window in hours (default: 24)'
    )
    parser.add_argument(
        '--step', type=float, default=60.0,
        help='Time step in seconds (default: 60)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SGP4 Orbit Propagator — Shell 3 (~550 km)")
    print("=" * 60)

    df = propagate_all(TLE_PATH, args.hours, args.step)

    print_summary(df)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  State vectors saved to: {OUTPUT_CSV}")

    print("\nGenerating ground track plot (all objects)...")
    plot_ground_tracks(df, OUTPUT_PLOT)

    # ------------------------------------------------------------------
    # Post-propagation altitude filter: keep only mean alt 540-560 km
    # ------------------------------------------------------------------
    df_filtered = filter_by_mean_altitude(df)

    print("\nGenerating ground track plot (filtered 540-560 km)...")
    plot_ground_tracks(
        df_filtered,
        OUTPUT_PLOT_FILTERED,
        title_suffix=f"  —  filtered {ALT_FILTER_MIN_KM:.0f}–{ALT_FILTER_MAX_KM:.0f} km",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
