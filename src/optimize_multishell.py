"""
optimize_multishell.py
----------------------
Single-shell (53 deg / 550 km) QUBO optimisation combining collision
avoidance (Pc) and population-weighted coverage.  Compares three solvers:
Random baseline, SA (dwave-neal), SQA (dimod).

Input:  data/multishell_pc.csv   (filtered to inc ~53 deg at load time)

QUBO parameters:
    N = candidates after single-shell filter
    k = K_SELECT = 100   (fixed: select exactly 100 satellites)
    P = 10 * N           (penalty coefficient)
    LAMBDA_COVERAGE = 0.7  (Pc weight; 1-lambda = coverage weight)

Objective:
    min [ lambda * sum_{i,j} w_ij x_i x_j  -  (1-lambda) * sum_i v_i x_i ]

QUBO matrix:
    Q[n,n] = P*(1-2k) - (1-lambda)*v_norm[n]    [cardinality + coverage]
    Q[n,m] = -w(n,m)  + 2*P                      [Pc avoidance + cardinality]
    w(n,m) = (1 - Pc_n) * (1 - Pc_m)

Coverage score:
    v_raw[i]  = exp( -(lat_i - 30)^2 / (2*20^2) )
    v_norm[i] = v_raw[i] / max(v_raw)

Outputs:
    results/multishell_comparison.csv
    results/multishell_SA_best.csv
    results/multishell_SQA_best.csv
    results/coverage_scores.csv      (if SAVE_COVERAGE_SCORES = True)

Usage:
    python src/optimize_multishell.py
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import dimod
except ImportError:
    dimod = None  # type: ignore[assignment]

try:
    import neal
except ImportError:
    neal = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
INPUT_CSV   = DATA_DIR / "multishell_pc.csv"

N_RUNS = 50   # independent solver runs

# ---------------------------------------------------------------------------
# Single-shell filter  (53 deg / 550 km)
# ---------------------------------------------------------------------------
SINGLE_SHELL_INC_DEG = 53.0
INC_TOL_DEG          = 1.0    # +/-1 deg

# ---------------------------------------------------------------------------
# Cardinality constraint  (fixed: select exactly K_SELECT satellites)
# ---------------------------------------------------------------------------
K_SELECT = 100   # number of satellites to select from Shell A

# ---------------------------------------------------------------------------
# Coverage parameters
# ---------------------------------------------------------------------------
LAMBDA_COVERAGE      = 0.7    # 0 = coverage only, 1 = Pc only
MU_LAT               = 30.0   # deg -- population-centroid latitude
SIGMA_LAT            = 20.0   # deg -- Gaussian spread of coverage weight
SAVE_COVERAGE_SCORES = True   # write results/coverage_scores.csv


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------

def build_qubo(
    pc_values: np.ndarray,
    k: int,
    P: float,
    v_norm: np.ndarray | None = None,
    lambda_cov: float = 1.0,
) -> np.ndarray:
    """
    Build the N×N upper-triangular QUBO matrix Q.

    Objective:
        min [ lambda_cov * sum_{i,j} P_ij x_i x_j
              - (1 - lambda_cov) * sum_i v_norm_i x_i ]

    Diagonal:   Q[n,n] = P * (1 - 2*k)               [cardinality constraint]
                       - (1 - lambda_cov) * v_norm[n]  [coverage linear term]
    Off-diag:   Q[n,m] = -w(n,m) + 2*P   where w(n,m) = (1-Pc_n)*(1-Pc_m)

    Scaling note: Pc values are O(1e-5), v_norm is O(1) after normalisation.
    With lambda_cov=0.7 the coverage term (~0.3) intentionally dominates the
    Pc signal to break degeneracy when Pc values are nearly uniform.
    """
    N = len(pc_values)
    Q = np.zeros((N, N), dtype=np.float64)

    diag_base = P * (1 - 2 * k)
    for n in range(N):
        Q[n, n] = diag_base
        if v_norm is not None:
            Q[n, n] -= (1.0 - lambda_cov) * v_norm[n]

    for n in range(N):
        for m in range(n + 1, N):
            w = (1.0 - pc_values[n]) * (1.0 - pc_values[m])
            Q[n, m] = -w + 2.0 * P

    return Q


# ---------------------------------------------------------------------------
# Aggregate Pc from a selection vector
# ---------------------------------------------------------------------------

def aggregate_pc(selected_indices: list[int], pc_values: np.ndarray) -> float:
    """Product formula: 1 - prod(1 - Pc_n) for selected satellites."""
    survival = 1.0
    for idx in selected_indices:
        survival *= (1.0 - pc_values[idx])
    return 1.0 - survival


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def run_random(pc_values: np.ndarray, k: int, n_runs: int) -> list[dict]:
    rng     = random.Random(0)
    N       = len(pc_values)
    results = []
    for run_i in range(n_runs):
        selected = rng.sample(range(N), k)
        results.append({
            'run':          run_i,
            'selected':     sorted(selected),
            'aggregate_pc': aggregate_pc(selected, pc_values),
        })
    return results


# ---------------------------------------------------------------------------
# SA solver (dwave-neal)
# ---------------------------------------------------------------------------

def run_sa(
    Q: np.ndarray,
    pc_values: np.ndarray,
    k: int,
    n_runs: int,
) -> list[dict]:
    if neal is None:
        print("  WARNING: neal not installed; SA skipped.")
        return []
    if dimod is None:
        print("  WARNING: dimod not installed; SA skipped.")
        return []

    N   = Q.shape[0]
    bqm = dimod.BinaryQuadraticModel('BINARY')
    for n in range(N):
        bqm.add_variable(n, Q[n, n])
    for n in range(N):
        for m in range(n + 1, N):
            if Q[n, m] != 0.0:
                bqm.add_interaction(n, m, Q[n, m])

    sampler  = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=n_runs,
                              num_sweeps=1000, seed=42)

    results = []
    for run_i, sample_rec in enumerate(response.samples()):
        selected = [n for n in range(N) if sample_rec.get(n, 0) == 1]
        results.append({
            'run':          run_i,
            'selected':     sorted(selected),
            'aggregate_pc': aggregate_pc(selected, pc_values),
        })
    return results


# ---------------------------------------------------------------------------
# SQA solver (dimod SimulatedAnnealingSampler)
# ---------------------------------------------------------------------------

def run_sqa(
    Q: np.ndarray,
    pc_values: np.ndarray,
    k: int,
    n_runs: int,
) -> list[dict]:
    if dimod is None:
        print("  WARNING: dimod not installed; SQA skipped.")
        return []

    N   = Q.shape[0]
    bqm = dimod.BinaryQuadraticModel('BINARY')
    for n in range(N):
        bqm.add_variable(n, Q[n, n])
    for n in range(N):
        for m in range(n + 1, N):
            if Q[n, m] != 0.0:
                bqm.add_interaction(n, m, Q[n, m])

    sampler  = dimod.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=n_runs,
                              num_sweeps=1000, seed=123)

    results = []
    for run_i, sample_rec in enumerate(response.samples()):
        selected = [n for n in range(N) if sample_rec.get(n, 0) == 1]
        results.append({
            'run':          run_i,
            'selected':     sorted(selected),
            'aggregate_pc': aggregate_pc(selected, pc_values),
        })
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def stats(results: list[dict]) -> dict:
    if not results:
        return {'best': float('nan'), 'mean': float('nan'), 'std': float('nan')}
    pcs  = [r['aggregate_pc'] for r in results]
    return {
        'best': float(min(pcs)),
        'mean': float(np.mean(pcs)),
        'std':  float(np.std(pcs)),
        'all':  pcs,
    }


def shell_distribution(
    results: list[dict],
    pc_values: np.ndarray,
    label_map: list[str],
) -> dict[str, int]:
    """Count shell distribution in the BEST solution."""
    if not results:
        return {"A": 0, "B": 0, "C": 0}
    pcs      = [r['aggregate_pc'] for r in results]
    best_idx = int(np.argmin(pcs))
    selected = results[best_idx]['selected']
    counts   = {"A": 0, "B": 0, "C": 0}
    for idx in selected:
        lbl = label_map[idx]
        if lbl in counts:
            counts[lbl] += 1
    return counts


def oom(pc_ref: float, pc_opt: float) -> str:
    """Orders of magnitude improvement: log10(pc_ref / pc_opt)."""
    if math.isnan(pc_ref) or math.isnan(pc_opt):
        return "N/A"
    if pc_opt <= 0.0:
        return "inf"
    if pc_ref <= 0.0:
        return "N/A"
    return f"{math.log10(pc_ref / pc_opt):+.2f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Single-Shell QUBO Optimisation  (Pc + Coverage)")
    print(f"  lambda = {LAMBDA_COVERAGE}  (Pc weight)  |  "
          f"1-lambda = {1-LAMBDA_COVERAGE:.1f}  (coverage weight)")
    print(f"  Coverage model: Gaussian(mu={MU_LAT}deg, sigma={SIGMA_LAT}deg)")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"\n  ERROR: {INPUT_CSV} not found.")
        print("  Run 'python src/compute_pc_multishell.py' first.")
        sys.exit(1)

    # --- Load and filter to single shell -----------------------------------
    df_all = pd.read_csv(INPUT_CSV)
    mask   = np.abs(df_all['inc_deg'].values - SINGLE_SHELL_INC_DEG) <= INC_TOL_DEG
    df     = df_all[mask].reset_index(drop=True)
    print(f"\n  Loaded {len(df_all)} candidates, kept {len(df)} "
          f"(inc {SINGLE_SHELL_INC_DEG}+/-{INC_TOL_DEG} deg filter)")

    N  = len(df)
    k  = K_SELECT
    P  = 10.0 * N

    if k > N:
        print(f"\n  ERROR: K_SELECT={k} > N={N}. "
              f"Increase N_PLANES_PER_SHELL in generate_multishell_candidates.py "
              f"to at least {k} before running.")
        sys.exit(1)

    pc_values  = df['Pc_n'].values.astype(np.float64)
    lat_values = df['lat_deg'].values.astype(np.float64)
    label_map  = df['shell_label'].tolist()
    norad_ids  = df['norad_id'].tolist()

    print(f"\n  N candidates : {N}")
    print(f"  k (select)   : {k}")
    print(f"  P (penalty)  : {P:.0f}")
    print(f"  Pc range     : {pc_values.min():.4e} - {pc_values.max():.4e}")
    print(f"  Pc mean      : {pc_values.mean():.4e}")
    print(f"  lat range    : {lat_values.min():.1f} - {lat_values.max():.1f} deg")
    print(f"  N_RUNS       : {N_RUNS}")

    # --- Coverage scores v_i -----------------------------------------------
    v_raw  = np.exp(-0.5 * ((lat_values - MU_LAT) / SIGMA_LAT) ** 2)
    v_norm = v_raw / v_raw.max()
    print(f"\n  Coverage scores (v_norm): "
          f"min={v_norm.min():.3f}  max={v_norm.max():.3f}  "
          f"mean={v_norm.mean():.3f}")

    if SAVE_COVERAGE_SCORES:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'sat_id':  norad_ids,
            'lat_deg': lat_values,
            'v_raw':   v_raw,
            'v_norm':  v_norm,
        }).to_csv(RESULTS_DIR / "coverage_scores.csv", index=False, float_format='%.6e')
        print(f"  Saved results/coverage_scores.csv")

    # --- Build QUBO ---------------------------------------------------------
    print("\nBuilding QUBO matrix (Pc + coverage)...")
    Q = build_qubo(pc_values, k, P, v_norm=v_norm, lambda_cov=LAMBDA_COVERAGE)
    print(f"  Q shape : {Q.shape}  |  non-zero entries : {int((Q != 0).sum())}")

    # --- Run solvers --------------------------------------------------------
    print(f"\n--- Random baseline ({N_RUNS} runs) ---")
    rand_results = run_random(pc_values, k, N_RUNS)
    print(f"  Best Pc = {stats(rand_results)['best']:.4e}")

    print(f"\n--- SA - neal.SimulatedAnnealingSampler ({N_RUNS} reads) ---")
    sa_results = run_sa(Q, pc_values, k, N_RUNS)
    if sa_results:
        print(f"  Best Pc = {stats(sa_results)['best']:.4e}")

    print(f"\n--- SQA - dimod.SimulatedAnnealingSampler ({N_RUNS} reads) ---")
    sqa_results = run_sqa(Q, pc_values, k, N_RUNS)
    if sqa_results:
        print(f"  Best Pc = {stats(sqa_results)['best']:.4e}")

    # --- Compute statistics -------------------------------------------------
    rand_s = stats(rand_results)
    sa_s   = stats(sa_results)
    sqa_s  = stats(sqa_results)

    rand_dist = shell_distribution(rand_results, pc_values, label_map)
    sa_dist   = shell_distribution(sa_results,   pc_values, label_map)
    sqa_dist  = shell_distribution(sqa_results,  pc_values, label_map)

    # --- Print results table ------------------------------------------------
    def _fmt(v: float) -> str:
        return f"{v:.4e}" if not math.isnan(v) else "N/A"

    w = 10
    print()
    print(f"  {'Metric':<22} {'Random':>{w}}  {'SA':>{w}}  {'SQA':>{w}}")
    print(f"  {'-'*22} {'-'*w}  {'-'*w}  {'-'*w}")
    print(f"  {'Best aggregate Pc':<22} {_fmt(rand_s['best']):>{w}}  "
          f"{_fmt(sa_s['best']):>{w}}  {_fmt(sqa_s['best']):>{w}}")
    print(f"  {'Mean aggregate Pc':<22} {_fmt(rand_s['mean']):>{w}}  "
          f"{_fmt(sa_s['mean']):>{w}}  {_fmt(sqa_s['mean']):>{w}}")
    print(f"  {'Std deviation':<22} {_fmt(rand_s['std']):>{w}}  "
          f"{_fmt(sa_s['std']):>{w}}  {_fmt(sqa_s['std']):>{w}}")
    print(f"  {'OOM vs random (best)':<22} {'-':>{w}}  "
          f"{oom(rand_s['best'], sa_s['best']):>{w}}  "
          f"{oom(rand_s['best'], sqa_s['best']):>{w}}")
    print(f"  {'Shell A selected':<22} {rand_dist['A']:>{w}}  "
          f"{sa_dist['A']:>{w}}  {sqa_dist['A']:>{w}}")
    print(f"  {'Shell B selected':<22} {rand_dist['B']:>{w}}  "
          f"{sa_dist['B']:>{w}}  {sqa_dist['B']:>{w}}")
    print(f"  {'Shell C selected':<22} {rand_dist['C']:>{w}}  "
          f"{sa_dist['C']:>{w}}  {sqa_dist['C']:>{w}}")

    # --- Gate ---------------------------------------------------------------
    best_opt = min(
        (sa_s['best']  if sa_results  else float('inf')),
        (sqa_s['best'] if sqa_results else float('inf')),
    )
    if best_opt < rand_s['best']:
        print(f"\n  GATE PASS: SA/SQA best ({best_opt:.4e}) < "
              f"random best ({rand_s['best']:.4e})  OK")
    else:
        print(f"\n  GATE: SA/SQA ({best_opt:.4e}) did not beat "
              f"random ({rand_s['best']:.4e}).")
        print("  This may occur when all Pc values are zero (SGP4 precision limit).")

    # --- Save outputs -------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Comparison CSV
    rows_cmp = []
    for solver, s, dist in [("Random", rand_s, rand_dist),
                             ("SA",     sa_s,   sa_dist),
                             ("SQA",    sqa_s,  sqa_dist)]:
        rows_cmp.append({
            'solver':            solver,
            'best_aggregate_pc': s['best'],
            'mean_aggregate_pc': s['mean'],
            'std_aggregate_pc':  s['std'],
            'shell_A_selected':  dist['A'],
            'shell_B_selected':  dist['B'],
            'shell_C_selected':  dist['C'],
        })
    pd.DataFrame(rows_cmp).to_csv(
        RESULTS_DIR / "multishell_comparison.csv", index=False, float_format='%.6e'
    )

    # Per-run aggregate Pc for convergence analysis (used by analyze_multishell.py)
    if rand_results and sa_results and sqa_results:
        df_runs = pd.DataFrame({
            'run':        list(range(N_RUNS)),
            'random_pc':  [r['aggregate_pc'] for r in rand_results[:N_RUNS]],
            'sa_pc':      [r['aggregate_pc'] for r in sa_results[:N_RUNS]],
            'sqa_pc':     [r['aggregate_pc'] for r in sqa_results[:N_RUNS]],
        })
        df_runs.to_csv(
            DATA_DIR / "multishell_runs.csv", index=False, float_format='%.6e'
        )

    # Best solutions
    def _save_best(results: list[dict], path: Path) -> None:
        if not results:
            return
        pcs      = [r['aggregate_pc'] for r in results]
        best_idx = int(np.argmin(pcs))
        selected = results[best_idx]['selected']
        rows     = []
        for idx in selected:
            rows.append({
                'norad_id':    norad_ids[idx],
                'shell_label': label_map[idx],
                'Pc_n':        pc_values[idx],
            })
        pd.DataFrame(rows).to_csv(path, index=False, float_format='%.6e')

    _save_best(sa_results,  RESULTS_DIR / "multishell_SA_best.csv")
    _save_best(sqa_results, RESULTS_DIR / "multishell_SQA_best.csv")

    print(f"\n  Saved results to:")
    print(f"    results/multishell_comparison.csv")
    print(f"    results/multishell_SA_best.csv")
    print(f"    results/multishell_SQA_best.csv")
    print(f"    data/multishell_runs.csv")
    print("Done.")


if __name__ == "__main__":
    main()
