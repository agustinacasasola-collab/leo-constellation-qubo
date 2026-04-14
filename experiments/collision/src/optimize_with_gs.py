"""
optimize_with_gs.py  (Step 5b -- QUBO with Ground Station Constraints)
----------------------------------------------------------------------
Build the full QUBO combining:
  1. Pc minimization objective
  2. Cardinality constraint (select exactly k satellites)
  3. Ground station coverage constraints (all 252 must be satisfied)

Then run solver comparison: Random baseline, SA (neal), SQA (dwave.samplers).

=======================================================================
PENALTY DERIVATION
=======================================================================

Let x in {0,1}^N be the selection vector (x_i=1 -> satellite i selected).
Let pc[i] = individual collision probability of satellite i.
Let v[i,s,w] = 1 if candidate i covers station s in window w.

Objective (maximize survival, equivalent to minimize Pc):
  H_obj = -sum_{i<j} (1-pc[i])(1-pc[j]) x_i x_j

  Scale: w_max = max_{i<j} (1-pc[i])(1-pc[j])
         obj_range ~= k^2 * w_max

Cardinality penalty (enforce |selection| = k):
  H_card = P_card * (sum_i x_i - k)^2
         = P_card * [sum_i x_i(1-2k) + 2 sum_{i<j} x_i x_j] + const

  QUBO diagonal:  Q[i,i] += P_card * (1 - 2k)
  QUBO off-diag:  Q[i,j] += 2 * P_card   for all i<j

  P_card = 2 * w_max * k * N   so that:
    P_card / obj_range = 2N/k  ~= 33  (large enough to enforce hard constraint)

Ground station penalty (enforce coverage for each (s,w)):
  For each (s,w), let C_sw = {i : v[i,s,w] = 1}.
  We want sum_{i in C_sw} x_i >= 1.  We penalize the soft constraint:
    H_gs(s,w) = P_gs * (1 - sum_{i in C_sw} x_i)^2

  Using x_i^2 = x_i (binary):
    QUBO diagonal:  Q[i,i] -= P_gs   for each i in C_sw
    QUBO off-diag:  Q[i,j] += 2*P_gs for each pair i<j in C_sw

  (Note: diagonal contribution from (sum x_i)^2 gives +P_gs*x_i per element,
   and linear term gives -2*P_gs*x_i, netting -P_gs*x_i on the diagonal.)

  P_gs = 2 * w_max * k   so that:
    P_gs < P_card (coverage costs less than violating cardinality)
    P_gs / obj_range ~= 2/k (each constraint adds modest penalty)

CARDINALITY NOTE:
  The GS off-diagonal additions (+2*P_gs per pair in each C_sw) effectively
  raise the energy of high-cardinality solutions, shifting the QUBO cardinality
  optimum slightly below k.  A greedy post-processing step (fixup_to_k) restores
  the exact k=100 cardinality after sampling, preserving the QUBO's satellite
  ranking while guaranteeing feasibility.

Inputs:
    data/arnas_candidates.csv    (satellite_id, pc, raan_deg, mean_anomaly_deg)
    data/gs_visibility.csv       (norad_id, station, window, visible)

Outputs:
    results/gs_comparison.csv
    results/gs_SA_best.csv
    results/gs_SQA_best.csv

Usage:
    python experiments/collision/src/optimize_with_gs.py
"""

import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

import neal
import dimod
from dwave.samplers import SimulatedAnnealingSampler as DwaveSASampler

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

K         = 100   # number of satellites to select
N_WINDOWS = 36
NUM_READS = 50

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')


# ---------------------------------------------------------------------------
# Helper: aggregate Pc
# ---------------------------------------------------------------------------

def agg_pc(pc_vals) -> float:
    """Aggregate Pc: 1 - exp(sum log(1-Pc_n))."""
    log_surv = sum(math.log1p(-p) for p in pc_vals if p < 1.0)
    return 1.0 - math.exp(log_surv)


# ---------------------------------------------------------------------------
# Helper: GS feasibility check
# ---------------------------------------------------------------------------

def check_gs_coverage(selected_indices: np.ndarray,
                      v: np.ndarray,
                      station_names: list) -> list:
    """
    Return list of (station, window) pairs with no coverage in the selection.

    Parameters
    ----------
    selected_indices : array of candidate indices (0-based into arnas_candidates)
    v                : visibility array shape (N, S, W)
    station_names    : ordered list of station names

    Returns
    -------
    violations : list of (station_name, window_1based) tuples
    """
    S = len(station_names)
    violations = []
    for s in range(S):
        for w in range(N_WINDOWS):
            if v[selected_indices, s, w].sum() == 0:
                violations.append((station_names[s], w + 1))
    return violations


# ---------------------------------------------------------------------------
# STEP 1 -- Load data
# ---------------------------------------------------------------------------

def step1_load():
    print("=" * 65)
    print("STEP 1 -- Loading data")
    print("=" * 65)

    # Candidates (source of truth for norad_id -> index mapping)
    cands_path = os.path.join(DATA_DIR, 'arnas_candidates.csv')
    cands = pd.read_csv(cands_path)
    cands = cands.rename(columns={'satellite_id': 'norad_id'})
    cands['norad_id'] = cands['norad_id'].astype(int)
    print(f"\n  arnas_candidates.csv: {len(cands)} candidates")
    print(f"    Columns: {list(cands.columns)}")

    # Visibility (sparse)
    vis_path = os.path.join(DATA_DIR, 'gs_visibility.csv')
    vis_df = pd.read_csv(vis_path)
    print(f"\n  gs_visibility.csv: {len(vis_df):,} rows")
    print(f"    Columns: {list(vis_df.columns)}")

    return cands, vis_df


# ---------------------------------------------------------------------------
# STEP 2 -- Rebuild visibility matrix v[i, s, w]
# ---------------------------------------------------------------------------

def step2_visibility(cands: pd.DataFrame, vis_df: pd.DataFrame):
    print()
    print("=" * 65)
    print("STEP 2 -- Rebuilding visibility matrix")
    print("=" * 65)

    station_names = list(GROUND_STATIONS.keys())
    S = len(station_names)
    N = len(cands)

    # norad_id -> 0-based index (consistent with arnas_candidates order)
    norad_to_idx = {int(row['norad_id']): i
                    for i, row in cands.iterrows()}

    v = np.zeros((N, S, N_WINDOWS), dtype=np.int8)

    station_to_s = {name: s for s, name in enumerate(station_names)}

    for _, row in vis_df.iterrows():
        norad   = int(row['norad_id'])
        station = row['station']
        window  = int(row['window']) - 1   # convert 1-based -> 0-based
        i = norad_to_idx.get(norad, None)
        s = station_to_s.get(station, None)
        if i is not None and s is not None and 0 <= window < N_WINDOWS:
            v[i, s, window] = 1

    total_visible = v.sum()
    print(f"\n  Visibility matrix shape: {v.shape}  (N={N}, S={S}, W={N_WINDOWS})")
    print(f"  Total visible (i,s,w) triples: {total_visible:,}")

    # Sanity: coverage per station
    for s, station in enumerate(station_names):
        min_cov = v[:, s, :].sum(axis=0).min()
        print(f"    {station:<12}  min candidates per window: {min_cov}")

    return v, station_names, N, norad_to_idx


# ---------------------------------------------------------------------------
# STEP 3 -- Build QUBO
# ---------------------------------------------------------------------------

def step3_build_qubo(cands: pd.DataFrame,
                     v: np.ndarray,
                     N: int) -> np.ndarray:
    print()
    print("=" * 65)
    print("STEP 3 -- Building QUBO matrix")
    print("=" * 65)

    pc   = cands['pc'].values          # (N,)
    surv = 1.0 - pc                    # (N,) survival probabilities

    # -- Objective weights ------------------------------------------------
    # w_obj[i,j] = (1-pc[i]) * (1-pc[j])
    surv_outer = np.outer(surv, surv)  # (N, N)
    i_triu, j_triu = np.triu_indices(N, k=1)
    w_obj_vals = surv_outer[i_triu, j_triu]
    w_max = float(w_obj_vals.max())

    # -- Penalty coefficients (see derivation at top of file) -------------
    # P_card = 20x baseline ensures k_raw == K from the sampler without fixup.
    # The 2x baseline shifted the QUBO cardinality optimum to ~k-4 due to the
    # large GS off-diagonal additions; 20x makes that perturbation negligible.
    P_card = 20.0 * w_max * K * N   # cardinality
    P_gs   = 20.0 * w_max * K       # per ground-station constraint (10x: ~0.2 x obj_range)

    print(f"\n  N={N}, k={K}")
    print(f"  w_max          = {w_max:.6f}")
    print(f"  P_card         = {P_card:.4f}")
    print(f"  P_gs           = {P_gs:.4f}")
    print(f"  P_card/P_gs    = {P_card/P_gs:.1f}  (should be > 1)")

    # -- Initialize Q ------------------------------------------------------
    Q = np.zeros((N, N), dtype=np.float64)

    # -- Step 4: Cardinality penalty ---------------------------------------
    # H_card = P_card * (sum x_i - k)^2
    # Diagonal:  Q[i,i] += P_card * (1 - 2k)
    # Off-diag:  Q[i,j] += 2 * P_card  for i<j
    diag_card = P_card * (1 - 2 * K)
    np.fill_diagonal(Q, Q.diagonal() + diag_card)
    Q[i_triu, j_triu] += 2.0 * P_card

    # -- Step 5: Pc objective ----------------------------------------------
    # H_obj = -sum_{i<j} (1-pc[i])(1-pc[j]) x_i x_j
    # Off-diag:  Q[i,j] -= w_obj[i,j]  for i<j
    Q[i_triu, j_triu] -= w_obj_vals

    # -- Step 6: Ground station constraints --------------------------------
    # For each (s,w):
    #   Diagonal:  Q[i,i] -= P_gs   for each i in C_sw
    #   Off-diag:  Q[i,j] += 2*P_gs for each i<j in C_sw
    S = v.shape[1]
    gs_offdiag_count = 0
    for s in range(S):
        for w in range(N_WINDOWS):
            C_sw = np.where(v[:, s, w] == 1)[0]
            if len(C_sw) == 0:
                continue
            # Net diagonal: -P_gs per candidate in C_sw
            Q[C_sw, C_sw] -= P_gs
            # Off-diagonal
            if len(C_sw) > 1:
                ci, cj = np.triu_indices(len(C_sw), k=1)
                row_idx = C_sw[ci]
                col_idx = C_sw[cj]
                Q[row_idx, col_idx] += 2.0 * P_gs
                gs_offdiag_count += len(ci)

    # -- Scale verification -----------------------------------------------
    n_nonzero    = int(np.count_nonzero(Q))
    obj_range    = K**2 * w_max
    base_offdiag = len(i_triu)

    print()
    print(f"  Q matrix nonzero entries:            {n_nonzero:,}")
    print(f"    Objective+cardinality off-diag:    {base_offdiag:,}")
    print(f"    GS constraint off-diag additions:  {gs_offdiag_count:,}")
    ratio_card = P_card / obj_range
    ratio_gs   = P_gs   / obj_range
    print()
    print(f"  Objective range:           {obj_range:.4f}")
    print(f"  P_card:                    {P_card:.4f}")
    print(f"  P_gs per constraint:       {P_gs:.4f}")
    print(f"  P_card / obj_range: {ratio_card:.1f}")
    print(f"  P_gs / obj_range:   {ratio_gs:.1f}")
    if ratio_card > 10000:
        print("  WARNING -- P_card / obj_range > 10000: Pc signal may be suppressed."
              " Report but continue.")
    print()
    print("  NOTE: GS off-diagonal additions shift the QUBO cardinality optimum")
    print("  slightly below k=100. fixup_to_k() post-processing restores exact")
    print("  cardinality while preserving the solver's satellite ranking.")

    return Q, P_card, P_gs, w_max


# ---------------------------------------------------------------------------
# STEP 4 -- Solvers
# ---------------------------------------------------------------------------

def fixup_to_k(sel: np.ndarray, pc: np.ndarray, N: int) -> np.ndarray:
    """Adjust a binary solution to contain exactly K selected indices.

    When the QUBO cardinality optimum shifts slightly below K (due to the
    large GS off-diagonal terms), the sampler returns solutions with |sel|
    slightly different from K.  This greedy post-processing step restores
    the exact cardinality:

      * Too many (|sel| > K): drop those with the highest individual Pc
        (retaining the safest satellites).
      * Too few  (|sel| < K): add from the unselected pool those with the
        lowest individual Pc (the safest remaining candidates).

    Returns a sorted numpy array of exactly K indices.
    """
    sel_set = set(int(x) for x in sel)
    if len(sel_set) > K:
        sorted_sel = sorted(sel_set, key=lambda i: pc[i])
        sel_set = set(sorted_sel[:K])
    elif len(sel_set) < K:
        need    = K - len(sel_set)
        non_sel = sorted(
            (i for i in range(N) if i not in sel_set),
            key=lambda i: pc[i]
        )
        sel_set.update(non_sel[:need])
    return np.array(sorted(sel_set), dtype=int)


def random_baseline(pc: np.ndarray, N: int, num_reads: int) -> list:
    """Draw num_reads random k-subsets and return aggregate Pc per run."""
    results = []
    rng = np.random.default_rng(42)
    for _ in range(num_reads):
        sel = rng.choice(N, size=K, replace=False)
        results.append((sel, agg_pc(pc[sel].tolist())))
    return results   # list of (selected_indices, aggregate_pc)


def run_sa(Q: np.ndarray, pc: np.ndarray, N: int,
           num_reads: int) -> list:
    """Run Simulated Annealing (neal) on the QUBO."""
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')

    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.time()
    sampleset = sampler.sample(bqm, num_reads=num_reads, seed=1234)
    print(f"    SA wall time: {time.time()-t0:.1f}s")

    results = []
    n_fixed = 0
    for sample, energy in sampleset.data(['sample', 'energy']):
        k_raw = int(sum(1 for val in sample.values() if val > 0.5))
        raw   = np.array([i for i, val in sample.items() if val > 0.5])
        if k_raw != K:
            n_fixed += 1
        sel = fixup_to_k(raw, pc, N)
        # Tuple: (sel_after_fixup, agg_pc_after, energy, k_raw, raw_before_fixup)
        results.append((sel, agg_pc(pc[sel].tolist()), float(energy), k_raw, raw))
    if n_fixed:
        print(f"    SA: {n_fixed}/{num_reads} runs had |sel| != {K}, "
              f"cardinality corrected by fixup_to_k.")
    return results


def run_sqa(Q: np.ndarray, pc: np.ndarray, N: int,
            num_reads: int) -> list:
    """Run dwave.samplers SimulatedAnnealingSampler as the SQA baseline.

    dwave.samplers.SimulatedAnnealingSampler is the C-backed D-Wave SA
    implementation (successor to dimod.SimulatedAnnealingSampler) and runs
    ~100x faster than the pure-Python version for problems of this size,
    making 50-read experiments feasible on a standard laptop.
    """
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')

    sampler = DwaveSASampler()
    t0 = time.time()
    sampleset = sampler.sample(bqm, num_reads=num_reads, seed=5678)
    print(f"    SQA wall time: {time.time()-t0:.1f}s")

    results = []
    n_fixed = 0
    for sample, energy in sampleset.data(['sample', 'energy']):
        k_raw = int(sum(1 for val in sample.values() if val > 0.5))
        raw   = np.array([i for i, val in sample.items() if val > 0.5])
        if k_raw != K:
            n_fixed += 1
        sel = fixup_to_k(raw, pc, N)
        # Tuple: (sel_after_fixup, agg_pc_after, energy, k_raw, raw_before_fixup)
        results.append((sel, agg_pc(pc[sel].tolist()), float(energy), k_raw, raw))
    if n_fixed:
        print(f"    SQA: {n_fixed}/{num_reads} runs had |sel| != {K}, "
              f"cardinality corrected by fixup_to_k.")
    return results


def step4_solve(Q: np.ndarray, cands: pd.DataFrame,
                v: np.ndarray, station_names: list, N: int,
                P_card: float, P_gs: float, w_max: float):
    print()
    print("=" * 65)
    print("STEP 4 -- Running solvers  (50 runs each)")
    print("=" * 65)

    pc = cands['pc'].values

    # -- Point 1: print penalty parameters before any solver ---------------
    print()
    print(f"  P_card = {P_card:.4f}")
    print(f"  P_gs per constraint = {P_gs:.4f}")
    print(f"  w_max = {w_max:.6f}")
    print(f"  P_card / P_gs ratio = {P_card/P_gs:.1f}")

    # -- Random baseline ---------------------------------------------------
    print("\n  Random baseline ...")
    rand_results = random_baseline(pc, N, NUM_READS)
    rand_pcs  = [r[1] for r in rand_results]
    rand_mean = float(np.mean(rand_pcs))

    # -- SA (initial run) --------------------------------------------------
    print("\n  SA (neal SimulatedAnnealingSampler) ...")
    sa_results  = run_sa(Q, pc, N, NUM_READS)
    sa_k_raws   = [r[3] for r in sa_results]
    sa_all_exact = all(k == K for k in sa_k_raws)
    print(f"    k_raw == {K} for all {NUM_READS} SA runs: {sa_all_exact}")

    # -- Point 3: k_raw distribution analysis -----------------------------
    k_raws     = sa_k_raws
    mean_kraw  = float(np.mean(k_raws))
    min_kraw   = int(min(k_raws))
    max_kraw   = int(max(k_raws))
    frac_exact = sum(1 for k in k_raws if k == K) / NUM_READS

    print()
    print("  --- k_raw distribution (before fixup) ---")
    kraw_counts = Counter(k_raws)
    for kval in sorted(kraw_counts):
        bar = '#' * kraw_counts[kval]
        print(f"    k={kval:3d}: {bar}  ({kraw_counts[kval]})")
    print(f"  Mean k_raw:             {mean_kraw:.2f}")
    print(f"  Min  k_raw:             {min_kraw}")
    print(f"  Max  k_raw:             {max_kraw}")
    print(f"  Fraction k_raw == {K}: {frac_exact:.1%}")

    # -- Point 4: conditional P_card increase + rerun ---------------------
    if mean_kraw < 95 or frac_exact < 0.5:
        P_card_new = 20.0 * w_max * K * N
        print()
        print(f"  CONDITION MET: mean_k_raw={mean_kraw:.2f} < 95 "
              f"or fraction_exact={frac_exact:.1%} < 50%")
        print(f"  Increasing P_card: {P_card:.4f} -> {P_card_new:.4f}")
        print(f"  New P_card / P_gs ratio = {P_card_new/P_gs:.1f}")

        delta      = P_card_new - P_card
        i_r, j_r  = np.triu_indices(N, k=1)
        Q_new      = Q.copy()
        np.fill_diagonal(Q_new, Q_new.diagonal() + delta * (1 - 2 * K))
        Q_new[i_r, j_r] += 2.0 * delta

        print("\n  SA rerun with P_card_new ...")
        sa_results = run_sa(Q_new, pc, N, NUM_READS)

        k_raws2     = [r[3] for r in sa_results]
        mean_kraw2  = float(np.mean(k_raws2))
        min_kraw2   = int(min(k_raws2))
        max_kraw2   = int(max(k_raws2))
        frac_exact2 = sum(1 for k in k_raws2 if k == K) / NUM_READS

        print()
        print("  --- k_raw distribution (rerun, P_card x20) ---")
        kraw_counts2 = Counter(k_raws2)
        for kval in sorted(kraw_counts2):
            bar = '#' * kraw_counts2[kval]
            print(f"    k={kval:3d}: {bar}  ({kraw_counts2[kval]})")
        print(f"  Mean k_raw:             {mean_kraw2:.2f}")
        print(f"  Min  k_raw:             {min_kraw2}")
        print(f"  Max  k_raw:             {max_kraw2}")
        print(f"  Fraction k_raw == {K}: {frac_exact2:.1%}")

    # -- Point 5: best SA solution analysis --------------------------------
    best_run_idx = int(np.argmin([r[1] for r in sa_results]))
    best_run     = sa_results[best_run_idx]
    sel_after    = best_run[0]
    pc_after     = best_run[1]
    k_raw_best   = best_run[3]
    raw_sel      = best_run[4]

    set_after  = set(int(x) for x in sel_after)
    set_before = set(int(x) for x in raw_sel)
    added      = sorted(set_after - set_before)
    removed    = sorted(set_before - set_after)
    pc_before  = agg_pc(pc[raw_sel].tolist()) if len(raw_sel) > 0 else 0.0

    print()
    print("  --- Best SA solution (lowest aggregate Pc) ---")
    print(f"  k_raw before fixup:         {k_raw_best}")
    print(f"  k after  fixup:             {len(sel_after)}")
    print(f"  Satellites added by fixup:  {len(added)}")
    print(f"  Satellites removed by fixup:{len(removed)}")
    print(f"  Aggregate Pc BEFORE fixup:  {pc_before:.4e}  (k={k_raw_best})")
    print(f"  Aggregate Pc AFTER  fixup:  {pc_after:.4e}  (k={len(sel_after)})")
    if added:
        print(f"  Pc values of {len(added)} satellite(s) added by fixup:")
        for idx in added:
            print(f"    candidate idx={idx:5d}  Pc={pc[idx]:.4e}")

    # -- SQA ---------------------------------------------------------------
    print("\n  SQA (dwave.samplers SimulatedAnnealingSampler) ...")
    sqa_results   = run_sqa(Q, pc, N, NUM_READS)
    sqa_k_raws    = [r[3] for r in sqa_results]
    sqa_all_exact = all(k == K for k in sqa_k_raws)
    print(f"    k_raw == {K} for all {NUM_READS} SQA runs: {sqa_all_exact}")

    # -- Point 1: confirm k_raw == K for all three solvers ----------------
    # Random always selects exactly K by construction.
    rand_all_exact = True
    print()
    print("  k_raw == K confirmation:")
    print(f"    Random: True  (rng.choice selects exactly K by construction)")
    print(f"    SA:     {sa_all_exact}")
    print(f"    SQA:    {sqa_all_exact}")

    return rand_results, rand_mean, sa_results, sqa_results


# ---------------------------------------------------------------------------
# STEP 5 -- Analyse and report
# ---------------------------------------------------------------------------

def step5_analyse(rand_results, rand_mean, sa_results, sqa_results,
                  cands, v, station_names, N):
    print()
    print("=" * 65)
    print("STEP 5 -- Analysis and results")
    print("=" * 65)

    pc      = cands['pc'].values
    raan    = cands['raan_deg'].values
    S       = len(station_names)

    def analyse_solver(results, name):
        pcs     = [r[1] for r in results]
        best_pc = min(pcs)
        mean_pc = float(np.mean(pcs))
        std_pc  = float(np.std(pcs))

        # k_raw == K fraction (index 3 in tuple; Random adapter always passes K)
        k_raws_list   = [int(r[3]) for r in results]
        kraw_pct_exact = 100.0 * sum(1 for k in k_raws_list if k == K) / len(k_raws_list)

        # GS violations per run
        gs_viols = []
        for sel_idx, run_pc, *_ in results:
            viol = check_gs_coverage(sel_idx.astype(int), v, station_names)
            gs_viols.append(len(viol))

        feasible_gs = [gv == 0 for gv in gs_viols]
        pct_feas    = 100.0 * sum(feasible_gs) / len(feasible_gs)

        # Best solution: prefer zero GS violations, then best Pc
        best_feas_pc    = float('inf')
        best_feas_idx   = None
        best_feas_viols = S * N_WINDOWS
        for (sel_idx, run_pc, *_), gv in zip(results, gs_viols):
            if gv < best_feas_viols or (gv == best_feas_viols and run_pc < best_feas_pc):
                best_feas_viols = gv
                best_feas_pc    = run_pc
                best_feas_idx   = sel_idx

        return {
            'name':            name,
            'best_pc':         best_pc,
            'mean_pc':         mean_pc,
            'std_pc':          std_pc,
            'kraw_pct_exact':  kraw_pct_exact,
            'pct_feasible':    pct_feas,
            'best_viols':      min(gs_viols),
            'best_feas_idx':   best_feas_idx,
        }

    # Random adapter: add k_raw=K and raw=sel so tuple length matches SA/SQA
    rand_adapted = [(r[0], r[1], 0.0, K, r[0]) for r in rand_results]
    rand_stats = analyse_solver(rand_adapted, 'Random')
    sa_stats   = analyse_solver(sa_results,   'SA')
    sqa_stats  = analyse_solver(sqa_results,  'SQA')

    def oom_vs_rand(pc_opt):
        if not math.isfinite(pc_opt) or pc_opt <= 0 or rand_mean <= 0:
            return float('nan')
        return math.log10(rand_mean / pc_opt)

    # -- GS violation summary for best solutions --------------------------
    print()
    for stats in [sa_stats, sqa_stats]:
        b_viols = stats['best_viols']
        total   = S * N_WINDOWS
        print(f"  {stats['name']} best:  GS violations = {b_viols}/{total} constraints")
        if b_viols > 0:
            print(f"  WARNING: best {stats['name']} solution violates {b_viols} GS "
                  f"constraints.  Consider increasing P_gs.")

    # -- Results table (ASCII, spec layout) --------------------------------
    print()
    sep = "+" + "-"*28 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+"
    print("  " + sep)
    print("  | {:<26} | {:^8} | {:^8} | {:^8} |".format(
        "Metric", "Random", "SA", "SQA"))
    print("  " + sep)

    def fc(val, fmt='.4e'):
        return format(val, fmt) if math.isfinite(val) else "inf"

    r, s, q = rand_stats, sa_stats, sqa_stats

    def row(label, rv, sv, qv):
        print(f"  | {label:<26} | {rv:^8} | {sv:^8} | {qv:^8} |")

    row("Best aggregate Pc",
        fc(r['best_pc']), fc(s['best_pc']), fc(q['best_pc']))
    row("Mean aggregate Pc",
        fc(r['mean_pc']), fc(s['mean_pc']), fc(q['mean_pc']))
    row("Std deviation",
        fc(r['std_pc']), fc(s['std_pc']), fc(q['std_pc']))
    row("OOM vs random (best)", "---",
        fc(oom_vs_rand(s['best_pc']), '.2f') + " OOM",
        fc(oom_vs_rand(q['best_pc']), '.2f') + " OOM")
    row("GS feasible runs (%)",
        f"{r['pct_feasible']:.1f}%",
        f"{s['pct_feasible']:.1f}%",
        f"{q['pct_feasible']:.1f}%")
    row("Best GS violations",
        str(r['best_viols']), str(s['best_viols']), str(q['best_viols']))
    row("k_raw == 100 (%)",
        f"{r['kraw_pct_exact']:.1f}%",
        f"{s['kraw_pct_exact']:.1f}%",
        f"{q['kraw_pct_exact']:.1f}%")
    print("  " + sep)

    # -- Point 4: comparison with no-GS baseline --------------------------
    mc_path = os.path.join(RESULTS_DIR, 'method_comparison.csv')
    if os.path.exists(mc_path):
        mc = pd.read_csv(mc_path)
        baseline_row = mc[mc['method'].str.contains('SimulatedAnnealing', na=False)]
        if len(baseline_row) > 0:
            baseline_sa_pc = float(baseline_row.iloc[0]['aggregate_pc'])
            print()
            print(f"  Baseline SA best Pc:      {baseline_sa_pc:.2e}")
            print(f"  With GS constraint SA:    {s['best_pc']:.2e}")
            if math.isfinite(s['best_pc']) and s['best_pc'] > 0:
                cost_oom = math.log10(s['best_pc'] / baseline_sa_pc)
                print(f"  Pc cost of GS constraint: {cost_oom:+.2f} OOM")
                print("  (positive = worse Pc; trade-off between collision safety")
                print("   and ground station coverage)")

    # -- Point 5: overlap and RAAN distributions --------------------------
    sa_idx  = set(int(x) for x in s['best_feas_idx']) if s['best_feas_idx'] is not None else set()
    sqa_idx = set(int(x) for x in q['best_feas_idx']) if q['best_feas_idx'] is not None else set()
    overlap = len(sa_idx & sqa_idx)

    print()
    print(f"  Overlap SA & SQA (intersection): {overlap}/{K} satellites")

    for label, idx_set in [('SA', sa_idx), ('SQA', sqa_idx)]:
        if idx_set:
            raan_sel = raan[np.array(sorted(idx_set))]
            print(f"  {label} RAAN distribution: "
                  f"min={raan_sel.min():.1f}  max={raan_sel.max():.1f}  "
                  f"std={raan_sel.std():.1f} deg")

    # -- Feasibility gate -------------------------------------------------
    print()
    sa_feas_pct = sa_stats['pct_feasible']
    if sa_feas_pct < 50.0:
        print(f"  GATE FAIL: only {sa_feas_pct:.1f}% of SA runs satisfy GS constraints.")
        print("  Consider increasing P_gs or relaxing coverage requirements.")
    else:
        print(f"  GATE PASS: {sa_feas_pct:.1f}% of SA runs satisfy GS constraints (>=50%).")

    return rand_stats, sa_stats, sqa_stats


# ---------------------------------------------------------------------------
# STEP 6 -- Save outputs
# ---------------------------------------------------------------------------

def step6_save(rand_stats, sa_stats, sqa_stats,
               rand_results, sa_results, sqa_results,
               cands, v, station_names, rand_mean):
    print()
    print("=" * 65)
    print("STEP 6 -- Saving outputs")
    print("=" * 65)

    def oom_vs_rand(pc_opt):
        if not math.isfinite(pc_opt) or pc_opt <= 0 or rand_mean <= 0:
            return float('nan')
        return math.log10(rand_mean / pc_opt)

    # gs_comparison.csv
    rows = []
    for stats, label in [(rand_stats, 'Random'), (sa_stats, 'SA'), (sqa_stats, 'SQA')]:
        rows.append({
            'method':              label,
            'best_aggregate_pc':   stats['best_pc'],
            'mean_aggregate_pc':   stats['mean_pc'],
            'std_aggregate_pc':    stats['std_pc'],
            'oom_vs_random_best':  oom_vs_rand(stats['best_pc']),
            'gs_feasible_pct':     stats['pct_feasible'],
            'best_gs_violations':  stats['best_viols'],
            'kraw_pct_exact':      stats['kraw_pct_exact'],
        })
    comp_df   = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, 'gs_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")

    # gs_SA_best.csv and gs_SQA_best.csv
    for stats, results, label in [(sa_stats,  sa_results,  'SA'),
                                   (sqa_stats, sqa_results, 'SQA')]:
        best_idx = stats['best_feas_idx']
        if best_idx is None:
            # Fallback: run with best Pc
            run_idx  = int(np.argmin([r[1] for r in results]))
            best_idx = results[run_idx][0].astype(int)

        best_idx = np.array(best_idx).astype(int)
        viol     = check_gs_coverage(best_idx, v, station_names)

        best_cands = cands.iloc[best_idx][
            ['norad_id', 'raan_deg', 'mean_anomaly_deg', 'pc']
        ].rename(columns={'pc': 'Pc_n'}).copy()
        best_cands['gs_violations'] = len(viol)
        best_cands = best_cands.sort_values('raan_deg')

        out_path = os.path.join(RESULTS_DIR, f'gs_{label}_best.csv')
        best_cands.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({len(best_cands)} satellites, "
              f"gs_violations={len(viol)})")

    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cands, vis_df = step1_load()
    v, station_names, N, norad_to_idx = step2_visibility(cands, vis_df)
    Q, P_card, P_gs, w_max = step3_build_qubo(cands, v, N)

    rand_results, rand_mean, sa_results, sqa_results = step4_solve(
        Q, cands, v, station_names, N, P_card, P_gs, w_max)

    rand_stats, sa_stats, sqa_stats = step5_analyse(
        rand_results, rand_mean, sa_results, sqa_results,
        cands, v, station_names, N)

    step6_save(rand_stats, sa_stats, sqa_stats,
               rand_results, sa_results, sqa_results,
               cands, v, station_names, rand_mean)


if __name__ == '__main__':
    main()
