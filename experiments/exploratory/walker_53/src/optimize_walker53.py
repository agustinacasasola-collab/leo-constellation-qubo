"""
optimize_walker53.py  (Step 5 -- Walker-53 Experiment)
-------------------------------------------------------
Build a full QUBO for Walker-53 satellite selection and run a
Random / SA / SQA solver comparison.

QUBO formulation
----------------
  Variables : x_i in {0,1}, i=0..N-1  (select satellite i)
  Objective : H_obj  = -sum_{i<j} w(i,j) * x_i * x_j   (quadratic)
  Cardinality: H_card = P_card * (sum_i x_i - k)^2
  GS penalty : H_gs   = sum_{sw} P_gs * (1 - sum_{i in C_sw} x_i)^2

  # Edge weight per Owens-Fahrner et al. (2025) Eq.1:
  # w(i,j) = (1 - Pc_i)(1 - Pc_j)
  # With sigma=0.1 km and sparse Walker candidates,
  # most Pc values are near zero, making w(i,j) ≈ 1
  # for all pairs. The landscape is flat -- this is
  # documented as a known limitation at N=648.

Penalty coefficients (w_max = max w(i,j) over all pairs):
    P_card = 20.0 * w_max * k * N
    P_gs   =  2.0 * w_max * k * N

GS constraint: quadratic per (s,w), correct binary expansion x_i^2=x_i:
    Q[i,i] -= 1 * P_gs   for each i in C_sw
    Q[i,j] += 2 * P_gs   for each pair i<j in C_sw

Infeasible runs (k_raw != k or GS violations) are DISCARDED -- no fixup.

Inputs:
    data/walker53_pc.csv
    data/walker53_visibility.csv

Outputs:
    results/walker53_comparison.csv
    results/walker53_SA_best.csv
    results/walker53_SQA_best.csv

Usage:
    python experiments/walker_53/src/optimize_walker53.py
"""

import os
import sys
import math
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(EXP_DIR, 'data')
RESULTS_DIR = os.path.join(EXP_DIR, 'results')

PC_CSV      = os.path.join(DATA_DIR, 'walker53_pc.csv')
VIS_CSV     = os.path.join(DATA_DIR, 'walker53_visibility.csv')
OUT_CMP     = os.path.join(RESULTS_DIR, 'walker53_comparison.csv')
OUT_SA      = os.path.join(RESULTS_DIR, 'walker53_SA_best.csv')
OUT_SQA     = os.path.join(RESULTS_DIR, 'walker53_SQA_best.csv')

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
K           = 100       # number of satellites to select
N_RUNS_SA   = 50        # SA runs
N_RUNS_SQA  = 50        # SQA runs
N_RUNS_RAND = 1000      # random baseline trials


# ---------------------------------------------------------------------------
# Helper: aggregate Pc  (product formula)
# ---------------------------------------------------------------------------

def agg_pc(pc_list: list[float]) -> float:
    """1 - prod(1 - pc_i)  computed in log space."""
    if not pc_list:
        return 0.0
    log_s = sum(math.log1p(-min(p, 1.0 - 1e-15)) for p in pc_list)
    return -math.expm1(log_s)


def count_gs_violations(sel_set: set, C: list[list[int]],
                         S: int, W: int) -> int:
    """Number of (station, window) pairs with zero coverage."""
    viols = 0
    for sw_idx, covering in enumerate(C):
        if not any(i in sel_set for i in covering):
            viols += 1
    return viols


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def step1_load() -> tuple[np.ndarray, list[int], list[list[int]], list[str], int, int]:
    """
    Returns:
        pc         : ndarray (N,) -- per-satellite Pc
        norad_ids  : list of N int NORAD IDs
        C          : list of S*W lists -- C[sw] = [i indices of satellites
                     visible for (station s, window w)]
        station_names
        S          : number of stations
        W          : number of windows
    """
    print('\n--- Step 1: Load data ---')

    if not os.path.exists(PC_CSV):
        print(f'  ERROR: {PC_CSV} not found. Run compute_pc_walker53.py first.')
        sys.exit(1)
    if not os.path.exists(VIS_CSV):
        print(f'  ERROR: {VIS_CSV} not found. Run compute_visibility_walker53.py first.')
        sys.exit(1)

    # Load Pc
    df_pc = pd.read_csv(PC_CSV)
    df_pc = df_pc.sort_values('norad_id').reset_index(drop=True)
    norad_ids = df_pc['norad_id'].tolist()
    pc        = df_pc['Pc'].values.astype(np.float64)
    N         = len(norad_ids)
    print(f'  Satellites (N): {N}')
    print(f'  Pc > 0: {(pc > 0).sum()}  /  Pc == 0: {(pc == 0).sum()}')

    # Load visibility
    df_vis    = pd.read_csv(VIS_CSV)
    station_names = sorted(df_vis['station'].unique().tolist())
    S = len(station_names)
    W = int(df_vis['window'].max())
    print(f'  Stations (S): {S}  |  Windows (W): {W}')

    # Build norad -> index map
    norad_to_idx = {nid: i for i, nid in enumerate(norad_ids)}

    # C[sw_idx] = list of satellite indices covering (s, w)
    C = []
    for s_idx, station in enumerate(station_names):
        sub_s = df_vis[df_vis['station'] == station]
        for w in range(1, W + 1):
            cands = sub_s[sub_s['window'] == w]['norad_id'].tolist()
            indices = [norad_to_idx[nid] for nid in cands
                       if nid in norad_to_idx]
            C.append(indices)

    # Coverage stats
    cov_counts = [len(c) for c in C]
    print(f'  Coverage per (s,w): min={min(cov_counts)}  '
          f'mean={sum(cov_counts)/len(cov_counts):.1f}  max={max(cov_counts)}')
    binding = sum(1 for cnt in cov_counts if cnt < K)
    print(f'  Binding constraints (coverage < k={K}): {binding} / {S*W}')

    return pc, norad_ids, C, station_names, S, W


# ---------------------------------------------------------------------------
# Step 3: Build QUBO  (Owens-Fahrner edge weights)
# ---------------------------------------------------------------------------

def step3_build_qubo(pc: np.ndarray,
                     C: list[list[int]], N: int,
                     ) -> tuple[np.ndarray, float, float]:
    """
    Construct the QUBO matrix Q (N x N, upper triangular convention).

    H = H_obj + H_card + H_gs

    H_obj  = -sum_{i<j} w(i,j) * x_i * x_j   (quadratic, Owens-Fahrner)
             # Edge weight per Owens-Fahrner et al. (2025) Eq.1:
             # w(i,j) = (1 - Pc_i)(1 - Pc_j)
             # With sigma=0.1 km and sparse Walker candidates,
             # most Pc values are near zero, making w(i,j) ≈ 1
             # for all pairs. The landscape is flat -- this is
             # documented as a known limitation at N=648.
             -> Q[i,j] -= w(i,j)   [i < j]

    H_card = P_card * (sum_i x_i - K)^2
             -> Q[i,i] += P_card * (1 - 2*K)   [binary simplification x^2=x]
             -> Q[i,j] += 2 * P_card            [i < j]

    H_gs   = sum_{sw} P_gs * (1 - sum_{i in C_sw} x_i)^2
             [QUADRATIC per (s,w) -- correct binary expansion x_i^2=x_i]
             -> Q[i,i] -= 1 * P_gs   for each i in C_sw
             -> Q[i,j] += 2 * P_gs   for each pair i<j in C_sw

    Returns (Q, P_card, P_gs).
    """
    print('\n--- Step 3: Build QUBO ---')

    # -- Owens-Fahrner edge weights ------------------------------------------
    survival = 1.0 - pc                             # (N,)
    w_max = float(survival.max() ** 2)              # max w(i,j) = (1-min_Pc)^2
    w_min_pair = float(survival.min() * survival.max())  # min non-trivial pair

    P_card = 20.0 * w_max * K * N
    P_gs   =  2.0 * w_max * K * N

    print(f'  K      = {K}')
    print(f'  N      = {N}')
    print(f'  P_card = {P_card:.3e}')
    print(f'  P_gs   = {P_gs:.3e}')

    Q = np.zeros((N, N), dtype=np.float64)

    # -- Objective: Owens-Fahrner quadratic edge weights ---------------------
    i_r, j_r = np.triu_indices(N, k=1)
    w_ij = survival[i_r] * survival[j_r]
    Q[i_r, j_r] -= w_ij

    obj_range_k2 = float(K ** 2) * w_max   # scale reference: k^2 * w_max

    # -- Pc distribution and landscape diagnostics ---------------------------
    n_zero = int((pc == 0).sum())
    nonzero_pc = pc[pc > 0]
    pc_min_nz = float(nonzero_pc.min()) if len(nonzero_pc) else float('nan')
    pc_max_nz = float(nonzero_pc.max()) if len(nonzero_pc) else float('nan')
    w_ij_min  = float(w_ij.min())
    w_ij_max  = float(w_ij.max())

    print(f'\n  Pc distribution:')
    print(f'    Pc=0 fraction:       {n_zero}/{N}')
    print(f'    Non-zero Pc range:   [{pc_min_nz:.3e}, {pc_max_nz:.3e}]')
    print(f'    w(i,j) range:        [{w_ij_min:.6f}, {w_ij_max:.6f}]')
    print(f'    w_max:               {w_max:.6f}')
    print(f'    obj_range (k²×w_max): {obj_range_k2:.4f}')

    if w_ij_max - w_ij_min < 1e-6:
        print()
        print('  WARNING: landscape is flat -- all edge weights are effectively')
        print('  equal. SA/SQA will not outperform random. Document as known')
        print('  limitation. Continue anyway -- GS constraint is still meaningful.')

    # -- Cardinality -----------------------------------------------------------
    np.fill_diagonal(Q, Q.diagonal() + P_card * (1.0 - 2.0 * K))
    Q[i_r, j_r] += 2.0 * P_card

    # -- GS constraint (QUADRATIC per (s,w), correct binary expansion) -------
    # H_gs(s,w) = P_gs * (1 - sum_{i in C_sw} x_i)^2
    # Binary expansion x_i^2=x_i gives:
    #   Q[i,i] -= 1 * P_gs   for each i in C_sw  (linear term)
    #   Q[i,j] += 2 * P_gs   for each pair i<j in C_sw (quadratic cross terms)
    t_gs = time.perf_counter()
    n_gs_pairs = 0
    count = np.zeros(N, dtype=np.float64)

    for sw_idx, covering in enumerate(C):
        n_sw = len(covering)
        if n_sw == 0:
            continue
        for i in covering:
            Q[i, i] -= 1.0 * P_gs
            count[i] += 1.0
        for a in range(n_sw):
            for b in range(a + 1, n_sw):
                ci, cj = covering[a], covering[b]
                if ci > cj:
                    ci, cj = cj, ci
                Q[ci, cj] += 2.0 * P_gs
                n_gs_pairs += 1

    gs_elapsed = time.perf_counter() - t_gs
    print(f'\n  GS off-diagonal pairs added: {n_gs_pairs:,}  ({gs_elapsed:.1f} s)')
    print(f'  GS coverage per satellite: '
          f'min={count.min():.0f}  mean={count.mean():.1f}  max={count.max():.0f}')

    return Q, P_card, P_gs


# ---------------------------------------------------------------------------
# Step 4: Solvers
# ---------------------------------------------------------------------------

def energy_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x)


def run_random(Q: np.ndarray, pc: np.ndarray, C: list[list[int]],
               N: int, S: int, W: int, n_runs: int,
               ) -> tuple[np.ndarray, float, float]:
    """Uniform random selection of exactly K satellites (baseline)."""
    print(f'\n  Random baseline ({n_runs} trials) ...')
    best_pc  = float('inf')
    best_sel = None
    all_pc   = []
    rng = np.random.default_rng(42)

    for _ in range(n_runs):
        sel = rng.choice(N, size=K, replace=False)
        sel_set = set(int(x) for x in sel)
        viols = count_gs_violations(sel_set, C, S, W)
        if viols > 0:
            continue   # discard infeasible
        a = agg_pc(pc[sel].tolist())
        all_pc.append(a)
        if a < best_pc:
            best_pc  = a
            best_sel = sel.copy()

    if not all_pc:
        print('  WARNING: all random runs were infeasible.')
        return np.arange(K), float('nan'), float('nan')

    mean_pc = float(np.mean(all_pc))
    print(f'  Feasible runs: {len(all_pc)} / {n_runs}')
    print(f'  Best aggregate Pc  : {best_pc:.3e}')
    print(f'  Mean aggregate Pc  : {mean_pc:.3e}')
    return best_sel, best_pc, mean_pc


def run_sa(Q: np.ndarray, pc: np.ndarray, C: list[list[int]],
           N: int, S: int, W: int, n_runs: int,
           ) -> list[tuple[np.ndarray, float, float, int]]:
    """
    Simulated Annealing via dwave.samplers (C-backed).

    Returns list of (sel, agg_pc, energy, k_raw) tuples for FEASIBLE runs only.
    """
    print(f'\n  SA ({n_runs} reads) ...')
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError:
        try:
            import neal
            SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
        except ImportError:
            print('  ERROR: dwave.samplers or neal not found. Skipping SA.')
            return []

    import dimod

    t0  = time.perf_counter()
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    sampler  = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=n_runs,
                              num_sweeps=10000, seed=12345)
    elapsed  = time.perf_counter() - t0
    print(f'  Elapsed: {elapsed:.1f} s')

    results = []
    k_raw_vals = []

    for sample, energy in response.data(['sample', 'energy']):
        raw = np.array([sample[i] for i in range(N)], dtype=int)
        k_raw = int(raw.sum())
        k_raw_vals.append(k_raw)

        if k_raw != K:
            continue   # discard wrong cardinality

        sel_set = set(int(i) for i in np.where(raw == 1)[0])
        viols = count_gs_violations(sel_set, C, S, W)
        if viols > 0:
            continue   # discard GS violations

        sel  = np.array(sorted(sel_set), dtype=int)
        a_pc = agg_pc(pc[sel].tolist())
        results.append((sel, a_pc, float(energy), k_raw))

    if k_raw_vals:
        print(f'  k_raw distribution: '
              f'min={min(k_raw_vals)}  mean={sum(k_raw_vals)/len(k_raw_vals):.1f}'
              f'  max={max(k_raw_vals)}')
        frac_exact = sum(1 for k in k_raw_vals if k == K) / len(k_raw_vals)
        print(f'  k_raw == {K}: {frac_exact*100:.0f}%  ({int(frac_exact*len(k_raw_vals))}/{len(k_raw_vals)} runs)')

    print(f'  Feasible runs (k_raw={K} and GS ok): {len(results)} / {n_runs}')
    return results


def run_sqa(Q: np.ndarray, pc: np.ndarray, C: list[list[int]],
            N: int, S: int, W: int, n_runs: int,
            ) -> list[tuple[np.ndarray, float, float, int]]:
    """
    Simulated Quantum Annealing via dwave.samplers SimulatedAnnealingSampler
    with quantum-tunnelling parameters (serves as SQA proxy).
    """
    print(f'\n  SQA ({n_runs} reads) ...')
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError:
        try:
            import neal
            SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler
        except ImportError:
            print('  ERROR: dwave.samplers or neal not found. Skipping SQA.')
            return []

    import dimod

    t0  = time.perf_counter()
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    sampler  = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=n_runs,
                              num_sweeps=20000, seed=99999,
                              beta_range=[0.01, 10.0])
    elapsed  = time.perf_counter() - t0
    print(f'  Elapsed: {elapsed:.1f} s')

    results  = []
    k_raw_vals = []

    for sample, energy in response.data(['sample', 'energy']):
        raw = np.array([sample[i] for i in range(N)], dtype=int)
        k_raw = int(raw.sum())
        k_raw_vals.append(k_raw)

        if k_raw != K:
            continue

        sel_set = set(int(i) for i in np.where(raw == 1)[0])
        viols = count_gs_violations(sel_set, C, S, W)
        if viols > 0:
            continue

        sel  = np.array(sorted(sel_set), dtype=int)
        a_pc = agg_pc(pc[sel].tolist())
        results.append((sel, a_pc, float(energy), k_raw))

    if k_raw_vals:
        print(f'  k_raw distribution: '
              f'min={min(k_raw_vals)}  mean={sum(k_raw_vals)/len(k_raw_vals):.1f}'
              f'  max={max(k_raw_vals)}')
        frac_exact = sum(1 for k in k_raw_vals if k == K) / len(k_raw_vals)
        print(f'  k_raw == {K}: {frac_exact*100:.0f}%  ({int(frac_exact*len(k_raw_vals))}/{len(k_raw_vals)} runs)')

    print(f'  Feasible runs (k_raw={K} and GS ok): {len(results)} / {n_runs}')
    return results


# ---------------------------------------------------------------------------
# Step 5: Analyse and save
# ---------------------------------------------------------------------------

def step5_analyse(rand_best_sel: np.ndarray, rand_best_pc: float,
                  rand_mean_pc: float,
                  sa_results: list, sqa_results: list,
                  pc: np.ndarray, norad_ids: list[int],
                  C: list[list[int]], S: int, W: int) -> None:
    print('\n--- Step 5: Results ---')

    # -- Extract best results ------------------------------------------------
    def summarise(results, label):
        if not results:
            return {'label': label, 'best_pc': float('nan'),
                    'mean_pc': float('nan'), 'std_pc': float('nan'),
                    'best_gs_viols': -1, 'gs_feasible_pct': 0.0,
                    'kraw_pct': 0.0, 'best_sel': None, 'all_pc': []}
        all_pc  = [r[1] for r in results]
        best_r  = min(results, key=lambda r: r[1])
        best_sel    = best_r[0]
        best_sel_set = set(int(x) for x in best_sel)
        best_gs = count_gs_violations(best_sel_set, C, S, W)
        return {
            'label': label,
            'best_pc': best_r[1],
            'mean_pc': float(np.mean(all_pc)),
            'std_pc':  float(np.std(all_pc)),
            'best_gs_viols': best_gs,
            'gs_feasible_pct': 100.0,   # all results are already GS-feasible
            'kraw_pct': 100.0,           # all results have k_raw == K
            'best_sel': best_sel,
            'all_pc': all_pc,
        }

    rand_sum = {'label': 'Random',
                'best_pc': rand_best_pc, 'mean_pc': rand_mean_pc,
                'std_pc': float('nan'), 'best_gs_viols': 0,
                'gs_feasible_pct': float('nan'), 'kraw_pct': 100.0,
                'best_sel': rand_best_sel, 'all_pc': []}
    sa_sum  = summarise(sa_results,  'SA')
    sqa_sum = summarise(sqa_results, 'SQA')

    # -- Print results table -------------------------------------------------
    print()
    col  = 28
    print(f"  {'Metric':<{col}}  {'Random':>12}  {'SA':>12}  {'SQA':>12}")
    print(f"  {'-'*col}  {'-'*12}  {'-'*12}  {'-'*12}")

    def row(label, r_val, sa_val, sqa_val, fmt='{:.3e}'):
        def fmtv(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return '     n/a'
            if isinstance(v, str):
                return f'{v:>12}'
            return fmt.format(v)
        print(f"  {label:<{col}}  {fmtv(r_val):>12}  "
              f"{fmtv(sa_val):>12}  {fmtv(sqa_val):>12}")

    row('Best aggregate Pc', rand_sum['best_pc'],
        sa_sum['best_pc'], sqa_sum['best_pc'])
    row('Mean aggregate Pc', rand_sum['mean_pc'],
        sa_sum['mean_pc'], sqa_sum['mean_pc'])
    row('Std aggregate Pc', rand_sum['std_pc'],
        sa_sum['std_pc'], sqa_sum['std_pc'])
    row('Feasible runs (%)', float('nan'),
        float(len(sa_results)), float(len(sqa_results)),
        fmt='{:.0f}')
    row('Best GS violations', rand_sum['best_gs_viols'],
        sa_sum['best_gs_viols'], sqa_sum['best_gs_viols'],
        fmt='{:d}')

    # OOM vs random
    def oom(best_pc, rand_pc):
        if math.isnan(best_pc) or math.isnan(rand_pc) or best_pc <= 0:
            return float('nan')
        return math.log10(rand_pc / best_pc)

    row('OOM vs random (best)',
        float('nan'),
        oom(sa_sum['best_pc'], rand_sum['best_pc']),
        oom(sqa_sum['best_pc'], rand_sum['best_pc']),
        fmt='{:.2f}')

    print()

    # -- SA & SQA overlap (if both have results) ----------------------------
    if sa_sum['best_sel'] is not None and sqa_sum['best_sel'] is not None:
        sa_idx  = set(int(x) for x in sa_sum['best_sel'])
        sqa_idx = set(int(x) for x in sqa_sum['best_sel'])
        overlap = len(sa_idx & sqa_idx)
        print(f'  Overlap SA and SQA (intersection): {overlap}/{K} satellites')

        # RAAN from TLE -- infer from norad_id (NORAD 95001 = plane 0, sat 0)
        # NORAD = 95001 + plane_i*9 + sat_j  -> RAAN = plane_i * 5 deg
        def norad_to_raan(nid):
            plane_i = (nid - 95001) // 9
            return plane_i * 5.0

        for label, idx_set in [('SA', sa_idx), ('SQA', sqa_idx)]:
            nids = [norad_ids[i] for i in sorted(idx_set)]
            raans = [norad_to_raan(nid) for nid in nids]
            raans = np.array(raans)
            print(f'  {label} RAAN distribution: '
                  f'min={raans.min():.1f}  max={raans.max():.1f}  '
                  f'std={raans.std():.1f} deg')

    # -- Save comparison CSV --------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cmp_rows = []
    for s in [rand_sum, sa_sum, sqa_sum]:
        cmp_rows.append({
            'method':           s['label'],
            'best_aggregate_pc': s['best_pc'],
            'mean_aggregate_pc': s['mean_pc'],
            'std_aggregate_pc':  s['std_pc'],
            'best_gs_violations': s['best_gs_viols'],
            'n_feasible_runs':   (len(sa_results) if s['label'] == 'SA'
                                  else len(sqa_results) if s['label'] == 'SQA'
                                  else float('nan')),
        })
    pd.DataFrame(cmp_rows).to_csv(OUT_CMP, index=False)
    print(f'\n  Saved: {OUT_CMP}')

    # -- Save best selections -------------------------------------------------
    for s, path in [(sa_sum, OUT_SA), (sqa_sum, OUT_SQA)]:
        if s['best_sel'] is not None:
            nids = [norad_ids[i] for i in sorted(s['best_sel'])]
            pcs  = [float(pc[i]) for i in sorted(s['best_sel'])]
            pd.DataFrame({'norad_id': nids, 'Pc': pcs}).to_csv(path, index=False)
            print(f'  Saved: {path}  ({len(nids)} satellites)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 65)
    print('STEP 5 -- QUBO Optimisation  (Walker-53)')
    print('=' * 65)

    pc, norad_ids, C, station_names, S, W = step1_load()
    N = len(norad_ids)

    Q, P_card, P_gs = step3_build_qubo(pc, C, N)

    print('\n--- Step 4: Solvers ---')
    print(f'  N={N}  K={K}  S={S}  W={W}')

    rand_best_sel, rand_best_pc, rand_mean_pc = run_random(
        Q, pc, C, N, S, W, N_RUNS_RAND)

    sa_results  = run_sa( Q, pc, C, N, S, W, N_RUNS_SA)
    sqa_results = run_sqa(Q, pc, C, N, S, W, N_RUNS_SQA)

    step5_analyse(rand_best_sel, rand_best_pc, rand_mean_pc,
                  sa_results, sqa_results,
                  pc, norad_ids, C, S, W)

    print('\nDone.')


if __name__ == '__main__':
    main()
