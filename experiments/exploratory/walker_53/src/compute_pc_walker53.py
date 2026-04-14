"""
compute_pc_walker53.py  (Step 3 -- Walker-53 Experiment)
---------------------------------------------------------
Compute per-satellite collision probability for each of the 648
Walker-53 satellites against the real LEO catalog.

Pipeline (same 4 steps as collision/src/compute_pc.py):
  Step 1  Apogee/perigee filter           reduce ~25k catalog objects
  Step 2  Screening volume filter         TCA proxy < 10 km (fallback 50 km)
  Step 3  Scalar miss distance            isotropic case: no conjunction-plane
                                          projection needed (sigma_x = sigma_z)
  Step 4  Chan 2D formula (scalar)        Pc per conjunction
  Step 5  Aggregate Pc                    product formula

Vectorised implementation -- inverted loop
------------------------------------------
Naive approach (OLD): for each catalog object j, broadcast against all
N=648 walkers -> creates (N, T, 3) = 67 MB array 16667 times -> slow.

This implementation inverts the loop:

  Propagate catalog batch B objects -> cat_pos (B, T, 3) -- stays in cache
  For each walker i (648 walkers):
      diff     = cat_pos - walker_pos[i]   (B, T, 3)  -- B << N
      sq_dists = (diff^2).sum(axis=-1)    (B, T)
      min_sq   = sq_dists.min(axis=1)     (B,)
      -> screen pairs and compute Pc

With B=200, cat_pos (B,T,3) = 20.8 MB fits in L3 cache.
walker_pos[i] is just 104 KB.  All allocations stay small.

Parameters:
    sigma   = 0.1 km   (100 m per axis, isotropic covariance)
    rho     = 0.01 km  (10 m combined hard-body radius)
    Screen  = 10 km primary, 50 km fallback

For isotropic covariance (sigma_x = sigma_z = sigma), the Chan 2D
formula reduces to the scalar form:
    Pc = exp(-d^2 / (2*sigma^2)) * (1 - exp(-rho^2 / (2*sigma^2)))
where d is the TCA distance.  This is mathematically identical to the
full conjunction-plane formula and avoids needing velocities.

Inputs:
    experiments/walker_53/data/propagated_walker53.csv
    experiments/walker_53/data/walker53.tle       (epoch only)
    data/leo_catalog.tle  (project root)

Output:
    experiments/walker_53/data/walker53_pc.csv
        Columns: norad_id, Pc, n_conjunctions, tca_min_km
"""

import math
import os
import sys
import time as _time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sgp4.api import Satrec, SatrecArray, jday

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):
        return it

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.dirname(os.path.abspath(__file__)))))
EXP_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(EXP_DIR, 'data')

WALKER_TLE    = os.path.join(DATA_DIR, 'walker53.tle')
WALKER_PROP   = os.path.join(DATA_DIR, 'propagated_walker53.csv')
CATALOG_TLE   = os.path.join(ROOT, 'data', 'leo_catalog.tle')
OUTPUT_CSV    = os.path.join(DATA_DIR, 'walker53_pc.csv')

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
SIGMA_KM          = 0.1      # 100 m per axis (isotropic)
HARD_BODY_KM      = 0.01     # 10 m combined hard-body radius
SCREENING_KM      = 10.0     # primary threshold
SCREENING_FALL_KM = 50.0     # fallback threshold
MARGIN_KM         = 200.0    # apogee/perigee altitude margin
EARTH_RADIUS_KM   = 6371.0
SIMULATION_DAYS   = 3
STEP_SECONDS      = 60.0
N_TIMESTEPS       = int(SIMULATION_DAYS * 86400 / STEP_SECONDS) + 1  # 4321
CAT_BATCH_SIZE    = 200      # catalog batch (fits in L3 cache at ~20 MB)

# Pre-compute Chan formula constants
_CHAN_U     = (HARD_BODY_KM / SIGMA_KM) ** 2           # (rho/sigma)^2
_CHAN_COEFF = 1.0 - math.exp(-_CHAN_U / 2.0)           # 1 - exp(-u/2)
_SQ_THRESH_PRI  = SCREENING_KM ** 2
_SQ_THRESH_FALL = SCREENING_FALL_KM ** 2


def chan_pc_scalar(d_km: float) -> float:
    """Chan 2D Pc for isotropic covariance (scalar TCA distance)."""
    return math.exp(-(d_km / SIGMA_KM) ** 2 / 2.0) * _CHAN_COEFF


# ---------------------------------------------------------------------------
# TLE helpers
# ---------------------------------------------------------------------------

def load_tles(path: str) -> list[dict]:
    with open(path) as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    sats = []
    for i in range(0, len(raw) - 1, 2):
        l1, l2 = raw[i], raw[i + 1]
        if not (l1.startswith('1 ') and l2.startswith('2 ')):
            continue
        norad_str = l1[2:7].strip()
        if not norad_str.isdigit():
            continue
        norad = int(norad_str)
        try:
            mm   = float(l2[52:63])
        except ValueError:
            continue
        mu  = 3.986004418e5
        n   = mm * 2 * math.pi / 86400.0
        a   = (mu / n ** 2) ** (1.0 / 3.0)
        ecc = float('0.' + l2[26:33])
        sats.append({'norad_id': norad, 'line1': l1, 'line2': l2,
                     'apogee_km':  a * (1 + ecc) - EARTH_RADIUS_KM,
                     'perigee_km': a * (1 - ecc) - EARTH_RADIUS_KM})
    return sats


def parse_tle_epoch(l1: str) -> datetime:
    y2   = int(l1[18:20])
    doy  = float(l1[20:32])
    year = (2000 + y2) if y2 < 57 else (1900 + y2)
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)


def build_jd_fr(start: datetime, dur_s: float, step_s: float):
    n = int(dur_s / step_s) + 1
    jd_l, fr_l = [], []
    for i in range(n):
        t = start + timedelta(seconds=i * step_s)
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_l.append(jd); fr_l.append(fr)
    return np.array(jd_l), np.array(fr_l)


def apogee_perigee_filter(catalog, cand_alt, margin=MARGIN_KM):
    lo, hi = cand_alt - margin, cand_alt + margin
    return [s for s in catalog if s['apogee_km'] >= lo and s['perigee_km'] <= hi]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 65)
    print('STEP 3 -- Compute Pc  (Walker-53 vs LEO Catalog)')
    print('=' * 65)

    for p in [WALKER_TLE, WALKER_PROP, CATALOG_TLE]:
        if not os.path.exists(p):
            print(f'\n  ERROR: {p} not found.'); sys.exit(1)

    # -- Epoch & time arrays -------------------------------------------------
    walker_tles  = load_tles(WALKER_TLE)
    N            = len(walker_tles)
    walker_norads = [s['norad_id'] for s in walker_tles]
    epoch        = parse_tle_epoch(walker_tles[0]['line1'])
    jd_arr, fr_arr = build_jd_fr(epoch, SIMULATION_DAYS * 86400.0, STEP_SECONDS)
    T            = len(jd_arr)
    print(f'\n  Walker satellites: {N}')
    print(f'  Epoch: {epoch.strftime("%Y-%m-%d %H:%M:%S")} UTC  '
          f'({T:,} timesteps @ {STEP_SECONDS:.0f} s)')

    # -- Load pre-propagated walker positions (N, T, 3) ----------------------
    print(f'\n  Loading walker positions from {os.path.basename(WALKER_PROP)}...')
    t0 = _time.perf_counter()
    df_prop = pd.read_csv(WALKER_PROP)
    walker_pos = (df_prop.sort_values(['norad_id', 'timestep'])
                         [['x_km', 'y_km', 'z_km']].values
                         .reshape(N, T, 3)
                         .astype(np.float64))   # (N, T, 3)
    print(f'  walker_pos shape: {walker_pos.shape}  ({_time.perf_counter()-t0:.1f} s)')

    # -- Load & filter catalog -----------------------------------------------
    print(f'\n  Loading catalog...')
    catalog_all  = load_tles(CATALOG_TLE)
    walker_alt   = walker_tles[0]['apogee_km']
    cat_filtered = apogee_perigee_filter(catalog_all, walker_alt)
    n_cat        = len(cat_filtered)
    print(f'  Catalog: {len(catalog_all):,} total  ->  {n_cat:,} after alt filter')
    print(f'  (altitude band {walker_alt-MARGIN_KM:.0f}'
          f' -- {walker_alt+MARGIN_KM:.0f} km)')

    # -- Accumulators --------------------------------------------------------
    # Primary (<10 km) and fallback (10-50 km) tracked separately.
    # At end: use primary if n_conj_pri > 0, else fallback.
    log_no_pri   = np.zeros(N, dtype=np.float64)
    n_pri        = np.zeros(N, dtype=np.int32)
    tca_pri      = np.full(N, np.inf)

    log_no_fall  = np.zeros(N, dtype=np.float64)
    n_fall       = np.zeros(N, dtype=np.int32)
    tca_fall     = np.full(N, np.inf)

    # -- Pre-allocate buffers (float32 halves memory bandwidth) ---------------
    walker_pos_f32 = walker_pos.astype(np.float32)          # (N, T, 3)
    diff_buf = np.empty((CAT_BATCH_SIZE, T, 3), dtype=np.float32)
    sq_buf   = np.empty((CAT_BATCH_SIZE, T),    dtype=np.float32)

    _SQ_THRESH_FALL_F32 = np.float32(_SQ_THRESH_FALL)

    # -- Vectorised loop: catalog batches, inner loop over walkers -----------
    n_batches = math.ceil(n_cat / CAT_BATCH_SIZE)
    pbar = _tqdm(range(0, n_cat, CAT_BATCH_SIZE), total=n_batches,
                 desc='Cat batch', unit='batch', ncols=72)
    t_start = _time.perf_counter()

    for bstart in pbar:
        bend  = min(bstart + CAT_BATCH_SIZE, n_cat)
        b_n   = bend - bstart
        b_sats = cat_filtered[bstart:bend]

        # Propagate catalog batch with SatrecArray
        satrecs = [Satrec.twoline2rv(s['line1'], s['line2']) for s in b_sats]
        sat_arr = SatrecArray(satrecs)
        e_b, r_b, _ = sat_arr.sgp4(jd_arr, fr_arr)
        cat_pos_f64 = np.asarray(r_b, dtype=np.float64)   # (b_n, T, 3)
        e_b_np      = np.asarray(e_b, dtype=np.int32)     # (b_n, T)

        # Mark invalid SGP4 timesteps with large distance (out of any screen)
        invalid = (e_b_np != 0)              # (b_n, T)
        cat_pos_f64[invalid, :] = 1.0e6     # set all 3 coords to 1e6 km

        cat_pos_f32 = cat_pos_f64.astype(np.float32)      # (b_n, T, 3)

        # ---- Inner loop over walkers (cat_pos_f32 stays cache-hot) ----
        for i in range(N):
            # In-place subtraction into pre-allocated buffer -- no heap alloc
            np.subtract(cat_pos_f32, walker_pos_f32[i], out=diff_buf[:b_n])
            np.einsum('ijk,ijk->ij', diff_buf[:b_n], diff_buf[:b_n],
                      out=sq_buf[:b_n])
            min_sq_f32 = sq_buf[:b_n].min(axis=1)          # (b_n,)

            # Screen: keep pairs within fallback threshold
            fall_mask = min_sq_f32 < _SQ_THRESH_FALL_F32
            if not fall_mask.any():
                continue

            for j in np.where(fall_mask)[0]:
                d = math.sqrt(float(min_sq_f32[j]))
                if d < _SQ_THRESH_PRI ** 0.5:           # primary
                    if d < tca_pri[i]:
                        tca_pri[i] = d
                    pc = chan_pc_scalar(d)
                    if pc > 0.0:
                        log_no_pri[i]  += math.log1p(-min(pc, 1.0 - 1e-15))
                        n_pri[i]       += 1
                else:                                    # fallback only
                    if d < tca_fall[i]:
                        tca_fall[i] = d
                    pc = chan_pc_scalar(d)
                    if pc > 0.0:
                        log_no_fall[i] += math.log1p(-min(pc, 1.0 - 1e-15))
                        n_fall[i]      += 1

        elapsed = _time.perf_counter() - t_start
        pbar.set_postfix({'pri': int(n_pri.sum()), 'elapsed': f'{elapsed:.0f}s'})

    # -- Merge primary / fallback --------------------------------------------
    use_fallback = (n_pri == 0)
    log_no_coll  = np.where(use_fallback, log_no_fall, log_no_pri)
    n_conj       = np.where(use_fallback, n_fall,      n_pri)
    tca_min      = np.where(use_fallback, tca_fall,    tca_pri)

    agg_pc  = -np.expm1(log_no_coll)
    tca_out = np.where(np.isinf(tca_min), -1.0, tca_min)

    # -- Save ----------------------------------------------------------------
    df_out = pd.DataFrame({'norad_id':       walker_norads,
                           'Pc':             agg_pc,
                           'n_conjunctions': n_conj,
                           'tca_min_km':     tca_out})
    df_out.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = _time.perf_counter() - t_start
    print(f'\n  Total Pc elapsed: {total_elapsed:.1f} s')
    print(f'  Saved: {OUTPUT_CSV}  ({len(df_out)} rows)')

    # -- Gate prints ---------------------------------------------------------
    nonzero = df_out[df_out['Pc'] > 0.0]
    n_fb    = int(use_fallback.sum())
    print()
    print('  Gate summary')
    print(f'    Total walkers    : {N}')
    print(f'    Pc > 0           : {len(nonzero)}')
    print(f'    Pc == 0          : {N - len(nonzero)}')
    print(f'    Used fallback    : {n_fb}  (walkers with no primary conjunction)')

    if len(nonzero) > 0:
        pv = nonzero['Pc'].values
        pc_min = float(pv.min()); pc_max = float(pv.max())
        dyn = pc_max / pc_min if pc_min > 0 else float('inf')
        print(f'    Min Pc           : {pc_min:.3e}')
        print(f'    Max Pc           : {pc_max:.3e}')
        print(f'    Dynamic range    : {dyn:.1e}x')
    else:
        print('  WARNING: All Pc values are zero.')
        print('  The 60-s SGP4 timestep may be too coarse to capture sub-km encounters.')
        print('  Consider sigma_km=1.0 for a less conservative covariance assumption.')

    print('\nDone.')


if __name__ == '__main__':
    main()
