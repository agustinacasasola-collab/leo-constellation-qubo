"""
compute_visibility.py  (Step 4b — Ground Station Visibility)
-------------------------------------------------------------
Compute the visibility matrix v[i, s, w] for all candidates,
ground stations, and 2-hour time windows.

Input:
    experiments/collision/data/propagated_candidates.csv
    (norad_id, timestep, x_km, y_km, z_km — ECEF positions)

Outputs:
    data/gs_visibility.csv       — sparse (visible=1 rows only)
        Columns: norad_id, station, window, visible
    data/gs_coverage_count.csv   — 252 rows (station × window)
        Columns: station, window, coverage_count

Feasibility gate:
    Exits with code 1 if any (station, window) pair has zero
    coverage, i.e., no candidate can cover that constraint.

Usage:
    python experiments/collision/src/compute_visibility.py
"""

import os
import sys
from math import radians, cos, sin, degrees

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

GROUND_STATIONS = {
    'Nairobi':   ( -1.29,  36.82),
    'Lagos':     (  6.45,   3.39),
    'Singapore': (  1.35, 103.82),
    'Mumbai':    ( 19.08,  72.88),
    'Lima':      (-12.05, -77.04),
    'Bogota':    (  4.71, -74.07),
    'Darwin':    (-12.46, 130.84),
}

MIN_ELEVATION_DEG = 5.0
T_GAP_HOURS       = 2.0
SIMULATION_DAYS   = 3
TIMESTEP_SEC      = 60
N_TIMESTEPS       = 4321   # 3 days × 86400 s / 60 s + 1
N_WINDOWS         = 36     # 3 days / 2 h
WINDOW_SIZE       = 120    # timesteps per window (2 h × 60 min/h)
R_E               = 6371.0 # km, spherical Earth

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gs_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return ECEF position (km) of a ground station on a spherical Earth."""
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    return R_E * np.array([
        cos(lat) * cos(lon),
        cos(lat) * sin(lon),
        sin(lat),
    ])


def window_slices(n_timesteps: int, window_size: int, n_windows: int):
    """Return list of (start, end) index pairs for each window."""
    slices = []
    for w in range(n_windows):
        start = w * window_size
        end   = start + window_size
        if w == n_windows - 1:
            end = n_timesteps   # absorb any remainder into last window
        slices.append((start, end))
    return slices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("STEP 4b — Ground Station Visibility")
    print("=" * 65)

    # ── Load propagated positions ────────────────────────────────────────────
    prop_path = os.path.join(DATA_DIR, 'propagated_candidates.csv')
    print(f"\nLoading: {prop_path}")
    prop = pd.read_csv(prop_path)
    print(f"  Rows: {len(prop):,}  Columns: {list(prop.columns)}")

    norad_ids = sorted(prop['norad_id'].unique())
    N = len(norad_ids)
    print(f"  Unique candidates (N): {N}")
    print(f"  Timesteps expected: {N_TIMESTEPS}  "
          f"(found: {prop['timestep'].nunique()})")

    # Build positions array (N, T, 3) — sorted by norad_id, then timestep
    prop_sorted = prop.sort_values(['norad_id', 'timestep'])
    positions = prop_sorted[['x_km', 'y_km', 'z_km']].values  # (N*T, 3)
    positions = positions.reshape(N, N_TIMESTEPS, 3)           # (N, T, 3)

    # ── Pre-compute window slices ────────────────────────────────────────────
    slices = window_slices(N_TIMESTEPS, WINDOW_SIZE, N_WINDOWS)
    last_w_size = slices[-1][1] - slices[-1][0]
    print(f"\n  Windows: {N_WINDOWS} × {WINDOW_SIZE} timesteps "
          f"(last window: {last_w_size} timesteps)")

    # ── Visibility computation — vectorized per station ─────────────────────
    station_names = list(GROUND_STATIONS.keys())
    S = len(station_names)

    # v[i, s, w] — boolean visibility matrix
    v = np.zeros((N, S, N_WINDOWS), dtype=np.int8)

    print(f"\nComputing elevation angles for {N} × {S} × {N_TIMESTEPS} = "
          f"{N*S*N_TIMESTEPS:,} combinations …")

    for s_idx, (station, (lat_deg, lon_deg)) in enumerate(GROUND_STATIONS.items()):
        # Ground station ECEF and unit normal (identical for spherical Earth)
        gs_pos  = gs_ecef(lat_deg, lon_deg)   # (3,)
        gs_norm = gs_pos / np.linalg.norm(gs_pos)

        # Vector from ground station to each satellite at each timestep
        # positions: (N, T, 3),  gs_pos: (3,) → broadcast over (N, T)
        rho = positions - gs_pos               # (N, T, 3)

        # Normalize rho along last axis
        rho_mag  = np.linalg.norm(rho, axis=2, keepdims=True)  # (N, T, 1)
        rho_norm = rho / rho_mag                                # (N, T, 3)

        # Elevation angle: arcsin(dot(rho_norm, gs_norm))
        # dot product along axis=2
        sin_el = np.einsum('ntk,k->nt', rho_norm, gs_norm)     # (N, T)
        el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))  # (N, T)

        # Binary visibility per timestep
        visible_ts = (el_deg >= MIN_ELEVATION_DEG).astype(np.int8)  # (N, T)

        # Aggregate into windows: v[i,s,w] = 1 if any timestep visible
        for w, (wstart, wend) in enumerate(slices):
            window_sum = visible_ts[:, wstart:wend].sum(axis=1)  # (N,)
            v[:, s_idx, w] = (window_sum >= 1).astype(np.int8)

        n_visible = v[:, s_idx, :].sum()
        print(f"  {station:<12}  visible (i,w) pairs: {n_visible:,} / {N*N_WINDOWS:,}")

    # ── Coverage count ───────────────────────────────────────────────────────
    # coverage_count[s, w] = number of candidates covering (s, w)
    coverage = v.sum(axis=0)  # (S, W)

    # ── Feasibility table ────────────────────────────────────────────────────
    print()
    print("  Feasibility table — coverage_count[s, w]")
    print()

    # Header
    col_w = 5
    header_windows = ''.join(f'W{w+1:02d}'.rjust(col_w) for w in range(N_WINDOWS))
    print(f"  {'Station':<12} | {header_windows} | {'min':>5} {'mean':>6} {'max':>5}")
    print(f"  {'-'*12}-+-{'-'*(col_w*N_WINDOWS)}-+-{'-'*5}-{'-'*6}-{'-'*5}")

    for s_idx, station in enumerate(station_names):
        row_vals = coverage[s_idx, :]
        row_str  = ''.join(str(v_).rjust(col_w) for v_ in row_vals)
        print(f"  {station:<12} | {row_str} | "
              f"{row_vals.min():>5} {row_vals.mean():>6.1f} {row_vals.max():>5}")

    print()

    # ── Gate check ───────────────────────────────────────────────────────────
    zero_pairs = [(station_names[s], w+1)
                  for s in range(S) for w in range(N_WINDOWS)
                  if coverage[s, w] == 0]

    thin_pairs = [(station_names[s], w+1, int(coverage[s, w]))
                  for s in range(S) for w in range(N_WINDOWS)
                  if 0 < coverage[s, w] < 10]

    if zero_pairs:
        print("  ERROR: the following (station, window) pairs have ZERO coverage:")
        for station, win in zero_pairs:
            print(f"    station={station}  window=W{win:02d}")
        print()
        print("  T_gap=2h is infeasible for this configuration.")
        print("  Increase T_GAP_HOURS or remove that station.")
        print()
        print("  Suggestion: try T_GAP_HOURS=4 (18 windows) or T_GAP_HOURS=6 (12 windows).")
        sys.exit(1)

    if thin_pairs:
        for station, win, cnt in thin_pairs:
            print(f"  WARNING: station {station} window W{win:02d} has only {cnt} candidates.")
        print("  Constraint may be very tight for k=100.")
        print()

    if not zero_pairs:
        total_constraints = S * N_WINDOWS
        if not thin_pairs:
            print(f"  GATE PASS: all {total_constraints} constraints are feasible.")
        else:
            print(f"  GATE PASS (with warnings): all {total_constraints} constraints "
                  f"have at least 1 candidate.")

    # ── Save gs_visibility.csv (sparse) ─────────────────────────────────────
    print()
    id_arr      = np.array(norad_ids)        # (N,)
    rows_list = []
    for s_idx, station in enumerate(station_names):
        for w in range(N_WINDOWS):
            cands_visible = np.where(v[:, s_idx, w] == 1)[0]
            for i in cands_visible:
                rows_list.append((int(id_arr[i]), station, w + 1, 1))

    vis_df = pd.DataFrame(rows_list, columns=['norad_id', 'station', 'window', 'visible'])
    vis_path = os.path.join(DATA_DIR, 'gs_visibility.csv')
    vis_df.to_csv(vis_path, index=False)
    print(f"  Saved: {vis_path}  ({len(vis_df):,} rows)")

    # ── Save gs_coverage_count.csv ───────────────────────────────────────────
    cov_rows = []
    for s_idx, station in enumerate(station_names):
        for w in range(N_WINDOWS):
            cov_rows.append({
                'station':        station,
                'window':         w + 1,
                'coverage_count': int(coverage[s_idx, w]),
            })
    cov_df   = pd.DataFrame(cov_rows)
    cov_path = os.path.join(DATA_DIR, 'gs_coverage_count.csv')
    cov_df.to_csv(cov_path, index=False)
    print(f"  Saved: {cov_path}  ({len(cov_df)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────────
    flat_cov = coverage.flatten()
    min_val  = int(flat_cov.min())
    mean_val = float(flat_cov.mean())
    max_val  = int(flat_cov.max())
    min_idx  = int(flat_cov.argmin())
    min_s    = station_names[min_idx // N_WINDOWS]
    min_w    = (min_idx % N_WINDOWS) + 1

    print()
    print("  Summary")
    print(f"    Ground stations:       {S}")
    print(f"    Time windows (2h):     {N_WINDOWS}")
    print(f"    Total constraints:     {S * N_WINDOWS}")
    print(f"    Min coverage count:    {min_val}  (station {min_s}, window W{min_w:02d})")
    print(f"    Mean coverage count:   {mean_val:.1f}")
    print(f"    Max coverage count:    {max_val}")
    feasibility = "FAIL" if zero_pairs else "PASS"
    print(f"    Feasibility:           {feasibility}")
    print()
    print("Done.")


if __name__ == '__main__':
    main()
