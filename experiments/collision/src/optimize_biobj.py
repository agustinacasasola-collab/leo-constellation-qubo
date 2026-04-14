"""
optimize_biobj.py  —  Bi-objective extension of the Arnas Shell 3 experiment
=============================================================================
Extends the Owens-Fahrner QUBO with a real per-satellite ground-station
coverage term computed from propagated_candidates.csv.

Edge weight  (Owens-Fahrner et al. 2025, Eq. 1):
    w(i,j) = x * (1 - Pc_i)(1 - Pc_j)  +  y * cov_i * cov_j

x/y calibration  (Ehrgott 2005, weighted-sum scalarization):
    x = 1.0  (fixed)
    y = mean_pc_term / mean_cov_term
    where mean_pc_term  = mean_pairs[(1-Pc_i)(1-Pc_j)]
          mean_cov_term = mean_pairs[cov_i * cov_j]
    This ensures both terms contribute equal mean weight across all
    N*(N-1)/2 candidate pairs before any ratio sweep is applied.

QUBO (matches run_hybrid.py convention, P = 20000):
    Q[(i,i)] = P * (1 - 2*k)
    Q[(i,j)] = -w(i,j) + 2*P    for i < j

Ground stations (9 total, including Cairo and Santiago added for biobj):
    Nairobi, Lagos, Singapore, Mumbai, Lima, Bogota, Darwin, Cairo, Santiago

Steps executed sequentially:
    A  —  Compute per-satellite coverage scores  (9 GS × 36 windows = 324 pairs)
    B  —  Calibrate x, y scaling factors
    C  —  x/y ratio sweep  (ratios: inf, 10, 1, 0.1, 0  |  20 SA + 20 SQA each)
    D  —  Main run at x/y = 1  (50 Random + 50 SA + 50 SQA)

Inputs:
    data/propagated_candidates.csv   (norad_id, timestep, x_km, y_km, z_km)
    data/arnas_candidates.csv        (satellite_id, pc, raan_deg, mean_anomaly_deg)

Outputs:
    data/gs_coverage_arnas.csv       (norad_id, cov_raw, cov_norm)
    results/biobj_sweep.csv
    results/biobj_comparison.csv
    results/biobj_SA_best.csv
    results/biobj_SQA_best.csv

Usage:
    python experiments/collision/src/optimize_biobj.py
"""

import math
import os
import sys
import time
from math import radians, cos, sin

import numpy as np
import pandas as pd

try:
    import dimod
except ImportError:
    dimod = None

try:
    import neal
except ImportError:
    neal = None

try:
    from dwave.samplers import SimulatedAnnealingSampler as DwaveSA
except ImportError:
    DwaveSA = None

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
    'Cairo':     ( 30.06,  31.24),
    'Santiago':  (-33.45, -70.67),
}

MIN_ELEVATION_DEG = 5.0
N_WINDOWS         = 36
WINDOW_SIZE       = 120
N_TIMESTEPS       = 4321
R_E               = 6371.0
TOTAL_PAIRS       = len(GROUND_STATIONS) * N_WINDOWS   # 9 × 36 = 324

K  = 100
P  = 20000   # cardinality penalty  (same as run_hybrid.py)

# Experiment 1 baselines (Pc-only QUBO, from main.py --data arnas --k 100)
EXP1_SA_BEST_PC  = 8.79e-5
EXP1_SQA_BEST_PC = 2.23e-5

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')

PROP_CSV    = os.path.join(DATA_DIR, 'propagated_candidates.csv')
ARNAS_CSV   = os.path.join(DATA_DIR, 'arnas_candidates.csv')
COV_CSV     = os.path.join(DATA_DIR, 'gs_coverage_arnas.csv')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def agg_pc(pc_vals) -> float:
    """1 - prod(1 - Pc_i), computed in log space for numerical stability."""
    log_s = sum(math.log1p(-min(p, 1.0 - 1e-15)) for p in pc_vals if p >= 0)
    return -math.expm1(log_s)


def gs_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    return R_E * np.array([cos(lat) * cos(lon),
                            cos(lat) * sin(lon),
                            sin(lat)])


def window_slices(n_ts, ws, nw):
    slices = []
    for w in range(nw):
        start = w * ws
        end   = start + ws
        if w == nw - 1:
            end = n_ts
        slices.append((start, end))
    return slices


def build_qubo(N, K_sel, P_pen, w_ij, i_r, j_r):
    """Return QUBO dict  Q[(i,i)] and Q[(i,j)] for i<j."""
    Q = {(i, i): float(P_pen * (1 - 2 * K_sel)) for i in range(N)}
    neg_w_plus_2P = (-w_ij + 2.0 * P_pen).tolist()
    for idx in range(len(i_r)):
        Q[(int(i_r[idx]), int(j_r[idx]))] = neg_w_plus_2P[idx]
    return Q


def run_solver_sa(bqm, n_reads, seed=42):
    """Run SA via neal.  Returns list of selected-index arrays."""
    if neal is None:
        print('  ERROR: neal not installed. Skipping SA.')
        return []
    sampler  = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=n_reads,
                               num_sweeps=1000, seed=seed)
    results = []
    for sample, energy in response.data(['sample', 'energy']):
        sel = np.array([i for i, v in sample.items() if v == 1], dtype=int)
        results.append(sel)
    return results


def run_solver_sqa(bqm, n_reads, seed=99999):
    """Run SQA proxy via dwave.samplers SA with quantum-tunnelling params."""
    if DwaveSA is None:
        if neal is None:
            print('  ERROR: neither dwave.samplers nor neal available. Skipping SQA.')
            return []
        # Fallback: neal with different params
        sampler  = neal.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=n_reads,
                                   num_sweeps=5000, seed=seed)
    else:
        sampler  = DwaveSA()
        response = sampler.sample(bqm, num_reads=n_reads,
                                   num_sweeps=20000, seed=seed,
                                   beta_range=[0.01, 10.0])
    results = []
    for sample, energy in response.data(['sample', 'energy']):
        sel = np.array([i for i, v in sample.items() if v == 1], dtype=int)
        results.append(sel)
    return results


def filter_feasible(raw_results, pc, cov, K_sel):
    """Keep only runs with exactly K satellites selected."""
    feasible = []
    for sel in raw_results:
        if len(sel) != K_sel:
            continue
        a_pc  = agg_pc(pc[sel].tolist())
        m_cov = float(cov[sel].mean())
        feasible.append((sel, a_pc, m_cov))
    return feasible


# ---------------------------------------------------------------------------
# STEP A — Compute per-satellite coverage scores
# ---------------------------------------------------------------------------

def step_a_coverage() -> pd.DataFrame:
    print('=' * 68)
    print('STEP A — Compute per-satellite coverage scores')
    print('=' * 68)

    if os.path.exists(COV_CSV):
        print(f'  Found {COV_CSV} — loading cached coverage.')
        df = pd.read_csv(COV_CSV)
        print(f'  Rows: {len(df)}  Columns: {list(df.columns)}')
        return df

    if not os.path.exists(PROP_CSV):
        print(f'  ERROR: {PROP_CSV} not found.')
        sys.exit(1)

    t0 = time.perf_counter()
    print(f'  Loading {PROP_CSV} ...')
    prop = pd.read_csv(PROP_CSV)
    print(f'  Rows: {len(prop):,}')

    norad_ids = sorted(prop['norad_id'].unique())
    N = len(norad_ids)
    print(f'  Candidates: {N}  |  Timesteps: {N_TIMESTEPS}')

    prop_sorted = prop.sort_values(['norad_id', 'timestep'])
    positions   = prop_sorted[['x_km', 'y_km', 'z_km']].values.reshape(N, N_TIMESTEPS, 3)

    slices        = window_slices(N_TIMESTEPS, WINDOW_SIZE, N_WINDOWS)
    station_names = list(GROUND_STATIONS.keys())
    S             = len(station_names)

    v = np.zeros((N, S, N_WINDOWS), dtype=np.int8)

    for s_idx, (station, (lat_deg, lon_deg)) in enumerate(GROUND_STATIONS.items()):
        gs_pos  = gs_ecef(lat_deg, lon_deg)
        gs_norm = gs_pos / np.linalg.norm(gs_pos)

        rho      = positions - gs_pos                          # (N, T, 3)
        rho_mag  = np.linalg.norm(rho, axis=2, keepdims=True)
        rho_norm = rho / rho_mag

        sin_el = np.einsum('ntk,k->nt', rho_norm, gs_norm)    # (N, T)
        el_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

        vis_ts = (el_deg >= MIN_ELEVATION_DEG).astype(np.int8)

        for w, (wstart, wend) in enumerate(slices):
            v[:, s_idx, w] = (vis_ts[:, wstart:wend].sum(axis=1) >= 1).astype(np.int8)

        print(f'  {station:<12}  visible (i,w) pairs: {v[:, s_idx, :].sum():,}')

    cov_counts = v.sum(axis=(1, 2))              # (N,)  total (s,w) pairs covered
    cov_raw    = cov_counts / TOTAL_PAIRS
    cov_norm   = cov_raw / cov_raw.max()

    df_cov = pd.DataFrame({'norad_id': norad_ids,
                            'cov_raw':  cov_raw,
                            'cov_norm': cov_norm})
    os.makedirs(DATA_DIR, exist_ok=True)
    df_cov.to_csv(COV_CSV, index=False)

    elapsed = time.perf_counter() - t0
    print(f'\n  Coverage diagnostic  (total (station, window) pairs = {TOTAL_PAIRS}):')
    print(f'    min cov_raw:    {cov_raw.min():.4f}  ({int(cov_counts.min())}/{TOTAL_PAIRS})')
    print(f'    max cov_raw:    {cov_raw.max():.4f}  ({int(cov_counts.max())}/{TOTAL_PAIRS})')
    print(f'    mean cov_raw:   {cov_raw.mean():.4f}')
    print(f'    std cov_raw:    {cov_raw.std():.4f}')
    ratio = cov_raw.max() / cov_raw.min() if cov_raw.min() > 0 else float('inf')
    print(f'    max/min ratio:  {ratio:.2f}')
    if ratio > 2.0:
        print('    DIFFERENTIATED — bi-objective is meaningful')
    elif ratio < 1.1:
        print('    WARNING — coverage is flat, bi-objective adds no signal')
    print(f'\n  Saved: {COV_CSV}  ({len(df_cov)} rows)  [{elapsed:.1f} s]')
    return df_cov


# ---------------------------------------------------------------------------
# STEP B — Calibrate x, y scaling factors
# ---------------------------------------------------------------------------

def step_b_calibrate(pc: np.ndarray, cov: np.ndarray):
    print()
    print('=' * 68)
    print('STEP B — Calibrate x/y scaling factors')
    print('=' * 68)

    N = len(pc)
    i_r, j_r = np.triu_indices(N, k=1)

    surv = 1.0 - pc
    pc_terms  = surv[i_r] * surv[j_r]    # (N*(N-1)/2,)
    cov_terms = cov[i_r]  * cov[j_r]

    mean_pc_term  = float(pc_terms.mean())
    mean_cov_term = float(cov_terms.mean())

    x = 1.0
    y = mean_pc_term / mean_cov_term if mean_cov_term > 0 else 1.0

    print(f'\n  N pairs: {len(i_r):,}')
    print(f'  mean_pc_term:   {mean_pc_term:.6f}')
    print(f'  mean_cov_term:  {mean_cov_term:.6f}')
    print(f'  x = {x:.4f}  (fixed)')
    print(f'  y = {y:.6f}  (calibrated for equal mean contribution)')
    balance_check = abs(x * mean_pc_term - y * mean_cov_term)
    print(f'  Balance: x*mean_pc_term = {x*mean_pc_term:.6f}  '
          f'y*mean_cov_term = {y*mean_cov_term:.6f}  '
          f'|diff| = {balance_check:.2e}')

    return x, y, mean_pc_term, mean_cov_term, i_r, j_r, surv, pc_terms, cov_terms


# ---------------------------------------------------------------------------
# STEP C — x/y ratio sweep
# ---------------------------------------------------------------------------

def step_c_sweep(pc, cov, surv, pc_terms, cov_terms,
                 mean_pc_term, mean_cov_term,
                 i_r, j_r, N):
    print()
    print('=' * 68)
    print('STEP C — x/y ratio sweep  (20 SA + 20 SQA per ratio)')
    print('=' * 68)

    if dimod is None:
        print('  ERROR: dimod not installed.')
        sys.exit(1)

    RATIOS = [float('inf'), 10.0, 1.0, 0.1, 0.0]
    total  = mean_pc_term + mean_cov_term

    sweep_rows = []

    ratio_xr_yr = {}   # store calibrated x_r, y_r per ratio for Step D

    for ratio in RATIOS:
        if ratio == float('inf'):
            x_r, y_r = 1.0, 0.0
        elif ratio == 0.0:
            x_r, y_r = 0.0, 1.0
        else:
            frac = ratio / (ratio + 1.0)
            x_r  = frac          * total / mean_pc_term  if mean_pc_term  > 0 else 0.0
            y_r  = (1.0 - frac)  * total / mean_cov_term if mean_cov_term > 0 else 0.0

        ratio_xr_yr[ratio] = (x_r, y_r)

        ratio_str = 'inf' if ratio == float('inf') else str(ratio)
        print(f'\n  ratio x/y = {ratio_str:>5}  '
              f'(x_r={x_r:.4f}, y_r={y_r:.4f})')

        w_ij = x_r * pc_terms + y_r * cov_terms
        w_max = float(w_ij.max())
        print(f'    w(i,j)  min={w_ij.min():.6f}  max={w_max:.6f}')

        t_build = time.perf_counter()
        Q   = build_qubo(N, K, P, w_ij, i_r, j_r)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        print(f'    QUBO built  [{time.perf_counter()-t_build:.1f} s]')

        # SA
        t0    = time.perf_counter()
        raw_sa = run_solver_sa(bqm, n_reads=20, seed=42)
        feas_sa = filter_feasible(raw_sa, pc, cov, K)
        print(f'    SA : {len(feas_sa):2d}/20 feasible  [{time.perf_counter()-t0:.1f} s]')

        # SQA
        t0     = time.perf_counter()
        raw_sqa = run_solver_sqa(bqm, n_reads=20, seed=99999)
        feas_sqa = filter_feasible(raw_sqa, pc, cov, K)
        print(f'    SQA: {len(feas_sqa):2d}/20 feasible  [{time.perf_counter()-t0:.1f} s]')

        sa_best_pc  = min((r[1] for r in feas_sa),  default=float('nan'))
        sa_mean_cov = float(np.mean([r[2] for r in feas_sa]))  if feas_sa  else float('nan')
        sqa_best_pc = min((r[1] for r in feas_sqa), default=float('nan'))
        sqa_mean_cov = float(np.mean([r[2] for r in feas_sqa])) if feas_sqa else float('nan')

        sweep_rows.append({
            'x_y_ratio':    ratio_str,
            'sa_best_pc':   sa_best_pc,
            'sa_mean_cov':  sa_mean_cov,
            'sa_feasible':  len(feas_sa),
            'sqa_best_pc':  sqa_best_pc,
            'sqa_mean_cov': sqa_mean_cov,
            'sqa_feasible': len(feas_sqa),
        })

    # Print sweep table
    print()
    print('  Sweep results:')
    hdr = (f"  {'x/y':>5}  {'SA best Pc':>14}  {'SA mean cov':>12}"
           f"  {'SQA best Pc':>14}  {'SQA mean cov':>12}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for row in sweep_rows:
        def fmt_pc(v):
            return f'{v:.3e}' if not math.isnan(v) else '       n/a'
        def fmt_cov(v):
            return f'{v:.4f}' if not math.isnan(v) else '      n/a'
        print(f"  {row['x_y_ratio']:>5}  {fmt_pc(row['sa_best_pc']):>14}"
              f"  {fmt_cov(row['sa_mean_cov']):>12}"
              f"  {fmt_pc(row['sqa_best_pc']):>14}"
              f"  {fmt_cov(row['sqa_mean_cov']):>12}")

    # Sanity check: x/y=inf SA best Pc within 0.5 OOM of EXP1_SA_BEST_PC
    inf_row = next(r for r in sweep_rows if r['x_y_ratio'] == 'inf')
    sa_inf  = inf_row['sa_best_pc']
    print()
    print(f'  SANITY CHECK (x/y=inf, Pc-only):')
    print(f'    SA best Pc:      {sa_inf:.3e}')
    print(f'    Experiment 1:    {EXP1_SA_BEST_PC:.3e}')
    if not math.isnan(sa_inf) and sa_inf > 0:
        oom_diff = abs(math.log10(sa_inf / EXP1_SA_BEST_PC))
        print(f'    |OOM difference|: {oom_diff:.2f}')
        if oom_diff <= 0.5:
            print('    PASS — within 0.5 OOM of Experiment 1.')
        else:
            print('    WARNING — QUBO differs from Experiment 1 (|OOM| > 0.5).')
    else:
        print('    WARNING — no feasible SA run at x/y=inf.')

    # Save sweep CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sweep_path = os.path.join(RESULTS_DIR, 'biobj_sweep.csv')
    pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
    print(f'\n  Saved: {sweep_path}')

    return ratio_xr_yr


# ---------------------------------------------------------------------------
# STEP D — Main run at x/y = 1  (50 runs each solver)
# ---------------------------------------------------------------------------

def step_d_main(pc, cov, surv, pc_terms, cov_terms,
                mean_pc_term, mean_cov_term,
                i_r, j_r, N, norad_ids, raan, mean_anom,
                ratio_xr_yr):
    print()
    print('=' * 68)
    print('STEP D — Main run at x/y = 1  (50 Random + 50 SA + 50 SQA)')
    print('=' * 68)

    x_r, y_r = ratio_xr_yr[1.0]
    print(f'\n  x_r = {x_r:.6f}  y_r = {y_r:.6f}')

    w_ij = x_r * pc_terms + y_r * cov_terms

    print('  Building QUBO ...')
    t0  = time.perf_counter()
    Q   = build_qubo(N, K, P, w_ij, i_r, j_r)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    print(f'  QUBO built  [{time.perf_counter()-t0:.1f} s]')

    rng = np.random.default_rng(42)

    # ---- Random baseline -----------------------------------------------
    print('\n  Random baseline (50 trials) ...')
    rand_feasible = []
    for _ in range(50):
        sel = rng.choice(N, size=K, replace=False)
        rand_feasible.append((sel,
                               agg_pc(pc[sel].tolist()),
                               float(cov[sel].mean())))

    rand_best_pc   = min(r[1] for r in rand_feasible)
    rand_mean_pc   = float(np.mean([r[1] for r in rand_feasible]))
    rand_best_cov  = max(r[2] for r in rand_feasible)
    rand_mean_cov  = float(np.mean([r[2] for r in rand_feasible]))
    print(f'    Feasible: 50/50  Best Pc: {rand_best_pc:.3e}  '
          f'Mean cov: {rand_mean_cov:.4f}')

    # ---- SA  -----------------------------------------------------------
    print('\n  SA (50 reads) ...')
    t0     = time.perf_counter()
    raw_sa = run_solver_sa(bqm, n_reads=50, seed=42)
    feas_sa = filter_feasible(raw_sa, pc, cov, K)
    print(f'  SA elapsed: {time.perf_counter()-t0:.1f} s  '
          f'Feasible: {len(feas_sa)}/50')

    # ---- SQA -----------------------------------------------------------
    print('\n  SQA (50 reads) ...')
    t0      = time.perf_counter()
    raw_sqa = run_solver_sqa(bqm, n_reads=50, seed=99999)
    feas_sqa = filter_feasible(raw_sqa, pc, cov, K)
    print(f'  SQA elapsed: {time.perf_counter()-t0:.1f} s  '
          f'Feasible: {len(feas_sqa)}/50')

    # ---- Summarise -----------------------------------------------------
    def summarise(feas, label):
        if not feas:
            return {'label': label, 'best_pc': float('nan'),
                    'mean_pc': float('nan'), 'best_cov': float('nan'),
                    'mean_cov': float('nan'), 'n_feas': 0, 'best_sel': None}
        best_r = min(feas, key=lambda r: r[1])
        return {'label': label,
                'best_pc':  best_r[1],
                'mean_pc':  float(np.mean([r[1] for r in feas])),
                'best_cov': max(r[2] for r in feas),
                'mean_cov': float(np.mean([r[2] for r in feas])),
                'n_feas':   len(feas),
                'best_sel': best_r[0]}

    rand_sum = {'label': 'Random', 'best_pc': rand_best_pc,
                'mean_pc': rand_mean_pc, 'best_cov': rand_best_cov,
                'mean_cov': rand_mean_cov, 'n_feas': 50, 'best_sel': None}
    sa_sum   = summarise(feas_sa,  'SA')
    sqa_sum  = summarise(feas_sqa, 'SQA')

    def oom_vs(solver_pc, ref_pc):
        if math.isnan(solver_pc) or math.isnan(ref_pc) or solver_pc <= 0:
            return float('nan')
        return math.log10(ref_pc / solver_pc)

    def cov_gain(solver_cov, ref_cov):
        if math.isnan(solver_cov) or math.isnan(ref_cov) or ref_cov == 0:
            return float('nan')
        return (solver_cov - ref_cov) / ref_cov * 100.0   # % gain

    # ---- Print results table -------------------------------------------
    W = 10
    print()
    print(f"  {'Metric':<28}  {'Random':>{W}}  {'SA':>{W}}  {'SQA':>{W}}")
    print(f"  {'-'*28}  {'-'*W}  {'-'*W}  {'-'*W}")

    def row(label, rv, sav, sqav, fmt='{:.3e}'):
        def f(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return '       n/a'
            if isinstance(v, str):
                return f'{v:>{W}}'
            return fmt.format(v)
        print(f"  {label:<28}  {f(rv):>{W}}  {f(sav):>{W}}  {f(sqav):>{W}}")

    row('Best aggregate Pc',
        rand_sum['best_pc'], sa_sum['best_pc'], sqa_sum['best_pc'])
    row('Mean aggregate Pc',
        rand_sum['mean_pc'], sa_sum['mean_pc'], sqa_sum['mean_pc'])
    row('Best mean cov_norm',
        rand_sum['best_cov'], sa_sum['best_cov'], sqa_sum['best_cov'],
        fmt='{:.4f}')
    row('Mean mean cov_norm',
        rand_sum['mean_cov'], sa_sum['mean_cov'], sqa_sum['mean_cov'],
        fmt='{:.4f}')
    row('OOM vs random (best Pc)',
        '—',
        oom_vs(sa_sum['best_pc'],  rand_sum['best_pc']),
        oom_vs(sqa_sum['best_pc'], rand_sum['best_pc']),
        fmt='{:.2f}')
    row('Cov gain vs random (%)',
        '—',
        cov_gain(sa_sum['mean_cov'],  rand_sum['mean_cov']),
        cov_gain(sqa_sum['mean_cov'], rand_sum['mean_cov']),
        fmt='{:.1f}')
    row('Feasible runs k=100 (%)',
        f'{rand_sum["n_feas"]}/50',
        f'{sa_sum["n_feas"]}/50',
        f'{sqa_sum["n_feas"]}/50',
        fmt='{}')

    # SA ∩ SQA overlap
    if sa_sum['best_sel'] is not None and sqa_sum['best_sel'] is not None:
        sa_set  = set(int(x) for x in sa_sum['best_sel'])
        sqa_set = set(int(x) for x in sqa_sum['best_sel'])
        overlap = len(sa_set & sqa_set)
        print(f'\n  SA+SQA overlap (best runs): {overlap}/{K} satellites')

    # Compare with Experiment 1
    print()
    print('  Comparison with Experiment 1 (Pc-only QUBO):')
    print(f'    Exp1 SA  best Pc:   {EXP1_SA_BEST_PC:.2e}  '
          f'Biobj SA  best Pc:  {sa_sum["best_pc"]:.3e}')
    print(f'    Exp1 SQA best Pc:   {EXP1_SQA_BEST_PC:.2e}  '
          f'Biobj SQA best Pc:  {sqa_sum["best_pc"]:.3e}')
    if not math.isnan(sa_sum['mean_cov']):
        print(f'    Coverage gain SA :  {cov_gain(sa_sum["mean_cov"],  rand_sum["mean_cov"]):.1f}% vs random mean cov')
    if not math.isnan(sqa_sum['mean_cov']):
        print(f'    Coverage gain SQA:  {cov_gain(sqa_sum["mean_cov"], rand_sum["mean_cov"]):.1f}% vs random mean cov')

    # ---- Save results CSVs ---------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmp_rows = []
    for s in [rand_sum, sa_sum, sqa_sum]:
        cmp_rows.append({'method':          s['label'],
                         'best_pc':         s['best_pc'],
                         'mean_pc':         s['mean_pc'],
                         'best_cov_norm':   s['best_cov'],
                         'mean_cov_norm':   s['mean_cov'],
                         'n_feasible':      s['n_feas']})
    cmp_path = os.path.join(RESULTS_DIR, 'biobj_comparison.csv')
    pd.DataFrame(cmp_rows).to_csv(cmp_path, index=False)
    print(f'\n  Saved: {cmp_path}')

    def save_best(s, path):
        if s['best_sel'] is None:
            return
        idx  = sorted(s['best_sel'])
        nids = [norad_ids[i] for i in idx]
        pd.DataFrame({'norad_id':         nids,
                      'raan_deg':         [raan[i]      for i in idx],
                      'mean_anomaly_deg': [mean_anom[i] for i in idx],
                      'Pc_n':             [float(pc[i]) for i in idx],
                      'cov_norm':         [float(cov[i]) for i in idx],
                      }).to_csv(path, index=False)
        print(f'  Saved: {path}  ({len(nids)} satellites)')

    save_best(sa_sum,  os.path.join(RESULTS_DIR, 'biobj_SA_best.csv'))
    save_best(sqa_sum, os.path.join(RESULTS_DIR, 'biobj_SQA_best.csv'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 68)
    print('optimize_biobj.py  —  Bi-objective Arnas Shell 3')
    print('=' * 68)

    if dimod is None:
        print('ERROR: dimod is required.  pip install dimod')
        sys.exit(1)

    # ---- STEP A: coverage -------------------------------------------------
    df_cov = step_a_coverage()

    # ---- Load arnas_candidates and join -----------------------------------
    print()
    print('  Loading arnas_candidates.csv ...')
    df_arnas = pd.read_csv(ARNAS_CSV)
    df_arnas['norad_id'] = df_arnas['satellite_id'].astype(int)
    df_arnas = df_arnas.sort_values('norad_id').reset_index(drop=True)

    df_merged = df_arnas.merge(df_cov[['norad_id', 'cov_norm']], on='norad_id', how='left')
    missing   = df_merged['cov_norm'].isna().sum()
    if missing > 0:
        print(f'  WARNING: {missing} candidates have no coverage entry — setting cov_norm=0.')
        df_merged['cov_norm'] = df_merged['cov_norm'].fillna(0.0)

    N         = len(df_merged)
    norad_ids = df_merged['norad_id'].tolist()
    pc        = df_merged['pc'].values.astype(np.float64)
    cov       = df_merged['cov_norm'].values.astype(np.float64)
    raan      = df_merged['raan_deg'].values
    mean_anom = df_merged['mean_anomaly_deg'].values
    surv      = 1.0 - pc

    print(f'  Merged: {N} satellites  '
          f'Pc>0: {(pc>0).sum()}  cov range: [{cov.min():.4f}, {cov.max():.4f}]')

    # ---- STEP B: calibrate ------------------------------------------------
    (x, y, mean_pc_term, mean_cov_term,
     i_r, j_r, surv, pc_terms, cov_terms) = step_b_calibrate(pc, cov)

    # ---- STEP C: sweep ----------------------------------------------------
    ratio_xr_yr = step_c_sweep(pc, cov, surv, pc_terms, cov_terms,
                                mean_pc_term, mean_cov_term,
                                i_r, j_r, N)

    # ---- STEP D: main run -------------------------------------------------
    step_d_main(pc, cov, surv, pc_terms, cov_terms,
                mean_pc_term, mean_cov_term,
                i_r, j_r, N, norad_ids, raan, mean_anom,
                ratio_xr_yr)

    print('\nDone.')


if __name__ == '__main__':
    main()
