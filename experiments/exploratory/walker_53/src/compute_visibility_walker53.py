"""
compute_visibility_walker53.py  (Step 4 -- Walker-53 Experiment)
----------------------------------------------------------------
Compute the visibility matrix v[i, s, w] for all 648 Walker-53
satellites, 11 ground stations, and 36 two-hour windows.

Input:
    experiments/walker_53/data/propagated_walker53.csv
        (norad_id, timestep, x_km, y_km, z_km -- GCRF/ECI positions)

Outputs:
    data/walker53_visibility.csv       -- sparse (visible=1 rows only)
        Columns: norad_id, station, window, visible
    data/walker53_coverage_count.csv   -- 396 rows (11 stations x 36 windows)
        Columns: station, window, coverage_count

Feasibility gate:
    Reports any (station, window) with coverage_count < k (k=100).
    Such pairs create genuinely binding QUBO constraints.
    Exits with code 1 if coverage_count == 0 for any pair.

Usage:
    python experiments/walker_53/src/compute_visibility_walker53.py
"""

import os
import sys
from math import radians, cos, sin

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ground stations (7 original + 4 new)
# ---------------------------------------------------------------------------
GROUND_STATIONS = {
    'Nairobi':   ( -1.29,   36.82),
    'Lagos':     (  6.45,    3.39),
    'Singapore': (  1.35,  103.82),
    'Mumbai':    ( 19.08,   72.88),
    'Lima':      (-12.05,  -77.04),
    'Bogota':    (  4.71,  -74.07),
    'Darwin':    (-12.46,  130.84),
    'Madrid':    ( 40.42,   -3.70),
    'Beijing':   ( 39.91,  116.39),
    'Ottawa':    ( 45.42,  -75.69),
    'London':    ( 51.51,   -0.13),
}

# ---------------------------------------------------------------------------
# Simulation parameters  (must match propagate_walker53.py)
# ---------------------------------------------------------------------------
MIN_ELEVATION_DEG = 5.0
SIMULATION_DAYS   = 3
TIMESTEP_SEC      = 60
N_TIMESTEPS       = 4321   # 3 days x 86400 s / 60 s + 1
N_WINDOWS         = 36     # 3 days / 2 h
WINDOW_SIZE       = 120    # timesteps per 2-hour window
R_E               = 6371.0 # km, spherical Earth
K_SELECT          = 100    # target selection count (for binding-constraint check)

ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(EXP_DIR, 'data')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gs_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return ECEF position (km) of a ground station on spherical Earth."""
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    return R_E * np.array([cos(lat) * cos(lon),
                            cos(lat) * sin(lon),
                            sin(lat)])


def window_slices(n_timesteps: int, window_size: int,
                  n_windows: int) -> list[tuple[int, int]]:
    slices = []
    for w in range(n_windows):
        start = w * window_size
        end   = start + window_size
        if w == n_windows - 1:
            end = n_timesteps
        slices.append((start, end))
    return slices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 65)
    print('STEP 4 -- Ground Station Visibility  (Walker-53)')
    print('=' * 65)

    prop_path = os.path.join(DATA_DIR, 'propagated_walker53.csv')
    if not os.path.exists(prop_path):
        print(f'\n  ERROR: {prop_path} not found.')
        print('  Run propagate_walker53.py first.')
        sys.exit(1)

    # -- Load positions -------------------------------------------------------
    print(f'\nLoading: {prop_path}')
    prop = pd.read_csv(prop_path)
    print(f'  Rows: {len(prop):,}  Columns: {list(prop.columns)}')

    norad_ids = sorted(prop['norad_id'].unique())
    N = len(norad_ids)
    T_found = prop['timestep'].nunique()
    print(f'  Unique satellites (N): {N}')
    print(f'  Timesteps expected: {N_TIMESTEPS}  (found: {T_found})')

    # Build positions array (N, T, 3) -- sorted by norad_id, then timestep
    prop_sorted = prop.sort_values(['norad_id', 'timestep'])
    positions   = prop_sorted[['x_km', 'y_km', 'z_km']].values
    positions   = positions.reshape(N, N_TIMESTEPS, 3)   # (N, T, 3)

    # -- Window slices --------------------------------------------------------
    slices = window_slices(N_TIMESTEPS, WINDOW_SIZE, N_WINDOWS)
    last_w_size = slices[-1][1] - slices[-1][0]
    print(f'\n  Windows: {N_WINDOWS} x {WINDOW_SIZE} timesteps '
          f'(last window: {last_w_size} timesteps)')

    # -- Visibility computation -----------------------------------------------
    station_names = list(GROUND_STATIONS.keys())
    S = len(station_names)

    v = np.zeros((N, S, N_WINDOWS), dtype=np.int8)

    print(f'\nComputing elevation angles for {N} x {S} x {N_TIMESTEPS} = '
          f'{N*S*N_TIMESTEPS:,} combinations ...')

    for s_idx, (station, (lat_deg, lon_deg)) in enumerate(GROUND_STATIONS.items()):
        gs_pos  = gs_ecef(lat_deg, lon_deg)
        gs_norm = gs_pos / np.linalg.norm(gs_pos)

        rho      = positions - gs_pos               # (N, T, 3)
        rho_mag  = np.linalg.norm(rho, axis=2, keepdims=True)
        rho_norm = rho / rho_mag                    # (N, T, 3)

        sin_el = np.einsum('ntk,k->nt', rho_norm, gs_norm)
        el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))  # (N, T)

        visible_ts = (el_deg >= MIN_ELEVATION_DEG).astype(np.int8)  # (N, T)

        for w, (wstart, wend) in enumerate(slices):
            window_sum = visible_ts[:, wstart:wend].sum(axis=1)  # (N,)
            v[:, s_idx, w] = (window_sum >= 1).astype(np.int8)

        n_visible = v[:, s_idx, :].sum()
        print(f'  {station:<12}  visible (i,w) pairs: {n_visible:,} / {N*N_WINDOWS:,}')

    # -- Coverage count -------------------------------------------------------
    coverage = v.sum(axis=0)   # (S, W)

    # -- Feasibility table ----------------------------------------------------
    print()
    print('  Feasibility table -- coverage_count[s, w]  (target k=100)')
    print()

    col_w = 5
    header_windows = ''.join(f'W{w+1:02d}'.rjust(col_w) for w in range(N_WINDOWS))
    print(f"  {'Station':<12} | {header_windows} | {'min':>5} {'mean':>6} {'max':>5}")
    print(f"  {'-'*12}-+-{'-'*(col_w*N_WINDOWS)}-+-{'-'*5}-{'-'*6}-{'-'*5}")

    for s_idx, station in enumerate(station_names):
        row_vals = coverage[s_idx, :]
        row_str  = ''.join(str(v_).rjust(col_w) for v_ in row_vals)
        print(f'  {station:<12} | {row_str} | '
              f'{row_vals.min():>5} {row_vals.mean():>6.1f} {row_vals.max():>5}')

    print()

    # -- Gate checks ----------------------------------------------------------
    zero_pairs = [(station_names[s], w+1)
                  for s in range(S) for w in range(N_WINDOWS)
                  if coverage[s, w] == 0]

    binding_pairs = [(station_names[s], w+1, int(coverage[s, w]))
                     for s in range(S) for w in range(N_WINDOWS)
                     if 0 < coverage[s, w] < K_SELECT]

    if zero_pairs:
        print('  ERROR: the following (station, window) pairs have ZERO coverage:')
        for station, win in zero_pairs:
            print(f'    station={station}  window=W{win:02d}')
        print('\n  T_gap=2h is infeasible for this configuration.')
        sys.exit(1)

    if binding_pairs:
        print(f'  BINDING constraints  (coverage_count < k={K_SELECT}):')
        for station, win, cnt in binding_pairs:
            print(f'    station={station:<12}  window=W{win:02d}  coverage={cnt}')
        print()

    total_constraints = S * N_WINDOWS
    n_binding = len(binding_pairs)
    if n_binding > 0:
        print(f'  GATE PASS -- {n_binding}/{total_constraints} constraints are '
              f'BINDING (coverage < k={K_SELECT}).')
        print(f'  The QUBO GS penalty will be active for these pairs.')
    else:
        print(f'  GATE PASS -- All {total_constraints} constraints have '
              f'coverage >= k={K_SELECT}.')
        print(f'  NOTE: No binding constraints -- GS penalty may not be active.')

    # -- Save walker53_visibility.csv (sparse) --------------------------------
    print()
    id_arr   = np.array(norad_ids)
    rows_list = []
    for s_idx, station in enumerate(station_names):
        for w in range(N_WINDOWS):
            for i in np.where(v[:, s_idx, w] == 1)[0]:
                rows_list.append((int(id_arr[i]), station, w + 1, 1))

    vis_df   = pd.DataFrame(rows_list,
                             columns=['norad_id', 'station', 'window', 'visible'])
    vis_path = os.path.join(DATA_DIR, 'walker53_visibility.csv')
    vis_df.to_csv(vis_path, index=False)
    print(f'  Saved: {vis_path}  ({len(vis_df):,} rows)')

    # -- Save walker53_coverage_count.csv -------------------------------------
    cov_rows = [{'station': station_names[s], 'window': w + 1,
                 'coverage_count': int(coverage[s, w])}
                for s in range(S) for w in range(N_WINDOWS)]
    cov_df   = pd.DataFrame(cov_rows)
    cov_path = os.path.join(DATA_DIR, 'walker53_coverage_count.csv')
    cov_df.to_csv(cov_path, index=False)
    print(f'  Saved: {cov_path}  ({len(cov_df)} rows)')

    # -- Summary --------------------------------------------------------------
    flat_cov = coverage.flatten()
    min_val  = int(flat_cov.min())
    mean_val = float(flat_cov.mean())
    max_val  = int(flat_cov.max())
    min_idx  = int(flat_cov.argmin())
    min_s    = station_names[min_idx // N_WINDOWS]
    min_w    = (min_idx % N_WINDOWS) + 1

    print()
    print('  Summary')
    print(f'    Ground stations:       {S}')
    print(f'    Time windows (2h):     {N_WINDOWS}')
    print(f'    Total constraints:     {total_constraints}')
    print(f'    Binding constraints:   {n_binding}  (coverage < k={K_SELECT})')
    print(f'    Min coverage:          {min_val}  '
          f'(station {min_s}, window W{min_w:02d})')
    print(f'    Mean coverage:         {mean_val:.1f}')
    print(f'    Max coverage:          {max_val}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
