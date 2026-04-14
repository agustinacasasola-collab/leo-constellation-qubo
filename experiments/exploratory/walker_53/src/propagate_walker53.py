"""
propagate_walker53.py  (Step 2 -- Walker-53 Experiment)
--------------------------------------------------------
Propagate all 648 Walker-53 satellites using SGP4 (SatrecArray
vectorised batch mode) over a 3-day simulation window.

Input:
    experiments/walker_53/data/walker53.tle

Output:
    experiments/walker_53/data/propagated_walker53.csv
        Columns : norad_id, timestep, x_km, y_km, z_km
        Rows    : 648 x 4321 = 2,800,008

Simulation parameters (matching collision experiment):
    Duration  : 3 days (259,200 s)
    Time step : 60 s  ->  4321 timesteps
    Frame     : GCRF (SGP4 output, approx ECI J2000)
    Epoch     : Anchored to the TLE epoch  (t=0 at first line-1 epoch)

Sanity check:
    All altitudes expected in 548 -- 552 km  (550 km nominal, circular).
"""

import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(EXP_DIR, 'data')

INPUT_TLE  = os.path.join(DATA_DIR, 'walker53.tle')
OUTPUT_CSV = os.path.join(DATA_DIR, 'propagated_walker53.csv')

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIMULATION_DAYS = 3
STEP_SECONDS    = 60.0
N_TIMESTEPS     = int(SIMULATION_DAYS * 86400 / STEP_SECONDS) + 1  # 4321
BATCH_SIZE      = 200   # satellites per SatrecArray call (~20 MB per batch)
EARTH_RADIUS_KM = 6371.0
ALT_NOMINAL_KM  = 550.0
ALT_TOL_KM      = 3.0   # expected band: 547 -- 553 km


# ---------------------------------------------------------------------------
# TLE helpers
# ---------------------------------------------------------------------------

def load_tles(tle_path: str) -> list[dict]:
    """Parse a 2-line TLE file; return list of {norad_id, line1, line2}."""
    with open(tle_path) as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    sats = []
    for i in range(0, len(raw) - 1, 2):
        l1, l2 = raw[i], raw[i + 1]
        if l1.startswith('1 ') and l2.startswith('2 '):
            sats.append({'norad_id': int(l1[2:7]),
                         'line1': l1, 'line2': l2})
    return sats


def parse_tle_epoch(line1: str) -> datetime:
    """Parse the epoch from TLE line 1 and return a UTC datetime."""
    y2   = int(line1[18:20])
    doy  = float(line1[20:32])
    year = (2000 + y2) if y2 < 57 else (1900 + y2)
    return (datetime(year, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=doy - 1.0))


def build_time_arrays(start: datetime, duration_s: float,
                      step_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (jd_array, fr_array) for all timesteps."""
    n = int(duration_s / step_s) + 1
    jd_list, fr_list = [], []
    for i in range(n):
        t = start + timedelta(seconds=i * step_s)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_list.append(jd)
        fr_list.append(fr)
    return np.array(jd_list), np.array(fr_list)


# ---------------------------------------------------------------------------
# Vectorised propagation
# ---------------------------------------------------------------------------

def propagate_walker53(tle_path: str, output_csv: str) -> dict:
    """
    Propagate all walker53 satellites using SatrecArray.

    Returns a dict of summary statistics.
    """
    sats   = load_tles(tle_path)
    N      = len(sats)
    T      = N_TIMESTEPS
    norads = np.array([s['norad_id'] for s in sats], dtype=np.int32)
    satrecs = [Satrec.twoline2rv(s['line1'], s['line2']) for s in sats]

    # Anchor time arrays to TLE epoch
    epoch = parse_tle_epoch(sats[0]['line1'])
    jd_array, fr_array = build_time_arrays(epoch, SIMULATION_DAYS * 86400.0,
                                           STEP_SECONDS)
    jd_array = jd_array.astype(np.float64)
    fr_array = fr_array.astype(np.float64)

    # Initialise output CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w') as f:
        f.write('norad_id,timestep,x_km,y_km,z_km\n')

    # Pre-allocate timestep tile
    ts_tile = np.tile(np.arange(T, dtype=np.int32), BATCH_SIZE)

    n_errors = 0
    alt_min  =  math.inf
    alt_max  = -math.inf
    t0 = time.perf_counter()

    n_batches = math.ceil(N / BATCH_SIZE)
    pbar = tqdm(range(0, N, BATCH_SIZE), total=n_batches,
                desc='Propagating', unit='batch', ncols=72)

    for bstart in pbar:
        bend  = min(bstart + BATCH_SIZE, N)
        b_n   = bend - bstart
        b_sat = SatrecArray(satrecs[bstart:bend])
        b_nor = norads[bstart:bend]

        e_b, r_b, _ = b_sat.sgp4(jd_array, fr_array)
        r_np = np.asarray(r_b, dtype=np.float64)   # (b_n, T, 3)
        e_np = np.asarray(e_b, dtype=np.int32)     # (b_n, T)

        n_errors += int((e_np != 0).sum())

        # Altitude stats (valid rows only)
        r_flat = r_np.reshape(-1, 3)
        e_flat = e_np.ravel()
        valid  = e_flat == 0
        if valid.any():
            alts = np.linalg.norm(r_flat[valid], axis=1) - EARTH_RADIUS_KM
            if alts.min() < alt_min:
                alt_min = float(alts.min())
            if alts.max() > alt_max:
                alt_max = float(alts.max())

        # Build DataFrame and append to CSV
        norad_col = np.repeat(b_nor, T)
        ts_col    = ts_tile[:b_n * T]
        df_batch  = pd.DataFrame({
            'norad_id': norad_col,
            'timestep': ts_col,
            'x_km':     r_np[:, :, 0].ravel(),
            'y_km':     r_np[:, :, 1].ravel(),
            'z_km':     r_np[:, :, 2].ravel(),
        })
        df_batch.to_csv(output_csv, mode='a', index=False, header=False,
                        float_format='%.4f')
        pbar.set_postfix({'sats': f'{bend}/{N}', 'errs': n_errors})

    elapsed = time.perf_counter() - t0
    return {
        'N': N, 'T': T, 'total_rows': N * T,
        'n_errors': n_errors,
        'alt_min': alt_min, 'alt_max': alt_max,
        'elapsed': elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 68)
    print('STEP 2 -- Propagate Walker-53 Orbits')
    print('=' * 68)

    if not os.path.exists(INPUT_TLE):
        print(f'\n  ERROR: {INPUT_TLE} not found.')
        print('  Run generate_walker53.py first.')
        sys.exit(1)

    from sgp4.api import SatrecArray  # noqa: F401 (verify import)
    print(f'\n  Input      : {INPUT_TLE}')
    print(f'  Duration   : {SIMULATION_DAYS} days  ({SIMULATION_DAYS*24:.0f} h)')
    print(f'  Timestep   : {STEP_SECONDS:.0f} s  ->  {N_TIMESTEPS:,} steps per satellite')
    print(f'  Batch size : {BATCH_SIZE} satellites per SatrecArray call')
    print(f'  Output     : {OUTPUT_CSV}')
    print()

    stats = propagate_walker53(INPUT_TLE, OUTPUT_CSV)

    # -- Sanity checks -------------------------------------------------------
    print()
    print('=' * 68)
    print('SANITY CHECKS')
    print('=' * 68)

    err_ok  = stats['n_errors'] == 0
    alt_lo  = ALT_NOMINAL_KM - ALT_TOL_KM
    alt_hi  = ALT_NOMINAL_KM + ALT_TOL_KM
    alt_ok  = (alt_lo <= stats['alt_min'] and stats['alt_max'] <= alt_hi)

    print(f"  SGP4 errors      : {stats['n_errors']}"
          f"  ({'PASS' if err_ok else 'FAIL -- unexpected errors'})")
    print(f"  Altitude range   : {stats['alt_min']:.1f} -- {stats['alt_max']:.1f} km"
          f"  ({'PASS' if alt_ok else 'WARN -- outside expected band'})"
          f"  (expected {alt_lo:.0f}--{alt_hi:.0f} km)")
    print(f"  Total rows       : {stats['total_rows']:,}  "
          f"({stats['N']:,} sats x {stats['T']:,} timesteps)")
    print(f"  Elapsed          : {stats['elapsed']:.1f} s")

    fsize_mb = os.path.getsize(OUTPUT_CSV) / (1024 ** 2)
    print(f"  File size        : {fsize_mb:.1f} MB")
    print('=' * 68)
    print('Done.')


if __name__ == '__main__':
    main()
