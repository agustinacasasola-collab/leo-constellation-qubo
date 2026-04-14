"""
optimize_coverage_shell_a.py  --  Task 2
------------------------------------------
Bi-objective QUBO optimisation for Shell A: minimise collision risk (Pc)
while maximising geographic coverage (20 degN -- 50 degN band).

Both objectives are fully quadratic (pairwise), consistent with the QUBO
binary-product structure:

    w(i,j) = lambda * (1-Pc_i)*(1-Pc_j)  +  (1-lambda) * cov_i * cov_j

QUBO matrix:
    Q[i,j] = -w(i,j) + 2*P           (off-diagonal, i < j)
    Q[i,i] = P*(1 - 2*k)             (diagonal: cardinality constraint)

Objective (minimised by annealer):
    E(x) = sum_{i<=j} Q[i,j] * x_i * x_j

Cardinality:
    k = N // 2  (select exactly half the candidates)
    P = 10 * N  (penalty coefficient, dominates off-diagonal terms)

Pc source:
    data/multishell_pc.csv filtered to inc ~53 deg, merged to Shell A
    candidates by raan_deg.  If data/shell_a_pc.csv exists it takes
    precedence (produced by a separate Pc computation step).

Coverage source:
    data/coverage_shell_a.csv  (produced by compute_coverage_shell_a.py)

Solvers:
    Random baseline  (50 independent uniform-random selections)
    SA  -- neal.SimulatedAnnealingSampler    (50 reads, 1000 sweeps)
    SQA -- dimod.SimulatedAnnealingSampler   (50 reads, 1000 sweeps)

Lambda sweep (20 runs each):
    lambda in [0.3, 0.5, 0.7, 0.9, 1.0]

Scale check:
    Warns if mean_cov_term / mean_pc_term > 100 or < 0.01 at lambda=0.5.

Outputs:
    results/coverage_shell_a_comparison.csv   (solver summary)
    results/coverage_shell_a_SA_best.csv      (best SA solution)
    results/coverage_shell_a_SQA_best.csv     (best SQA solution)
    results/coverage_shell_a_lambda_sweep.csv (aggregate Pc per lambda)
    data/coverage_shell_a_runs.csv            (per-run Pc for all 50 runs)

Usage:
    python src/optimize_coverage_shell_a.py
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

SHELL_A_PC_CSV   = DATA_DIR / "shell_a_pc.csv"          # preferred Pc source
MULTISHELL_PC_CSV = DATA_DIR / "multishell_pc.csv"       # fallback Pc source
COVERAGE_CSV     = DATA_DIR / "coverage_shell_a.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_RUNS         = 50
N_SWEEP_RUNS   = 20
LAMBDA_DEFAULT = 0.5
LAMBDA_SWEEP   = [0.3, 0.5, 0.7, 0.9, 1.0]

SHELL_A_INC = 53.0
INC_TOL     = 1.0


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------

def build_qubo(
    pc_values: np.ndarray,
    cov_norm: np.ndarray,
    k: int,
    P: float,
    lambda_val: float,
) -> np.ndarray:
    """
    Build N x N upper-triangular QUBO matrix.

    Off-diagonal:
        w(i,j) = lambda*(1-Pc_i)*(1-Pc_j) + (1-lambda)*cov_i*cov_j
        Q[i,j] = -w(i,j) + 2*P

    Diagonal:
        Q[i,i] = P*(1-2k)   (cardinality constraint only)
    """
    N = len(pc_values)
    Q = np.zeros((N, N), dtype=np.float64)

    diag_val = P * (1.0 - 2.0 * k)
    np.fill_diagonal(Q, diag_val)

    pc1  = 1.0 - pc_values     # shape (N,)
    cov  = cov_norm             # shape (N,)

    for n in range(N):
        for m in range(n + 1, N):
            w       = lambda_val * pc1[n] * pc1[m] + (1.0 - lambda_val) * cov[n] * cov[m]
            Q[n, m] = -w + 2.0 * P

    return Q


def scale_check(pc_values: np.ndarray, cov_norm: np.ndarray) -> None:
    """Print a warning if the two objective scales are very mismatched."""
    pc1        = 1.0 - pc_values
    mean_pc    = float(np.mean(pc1[:, None] * pc1[None, :]))
    mean_cov   = float(np.mean(cov_norm[:, None] * cov_norm[None, :]))
    if mean_pc == 0.0:
        print("  WARNING: all Pc values are 1.0 (no collision risk signal).")
        return
    ratio = mean_cov / mean_pc
    print(f"  Scale check (lambda=0.5): "
          f"mean_cov_term={mean_cov:.4e}  mean_pc_term={mean_pc:.4e}  "
          f"ratio={ratio:.2f}")
    if ratio > 100.0:
        print("  WARNING: coverage term >> Pc term (ratio > 100). "
              "Coverage dominates at lambda < 0.5.")
    elif ratio < 0.01:
        print("  WARNING: Pc term >> coverage term (ratio < 0.01). "
              "Pc dominates at lambda > 0.5.")


# ---------------------------------------------------------------------------
# Aggregate Pc
# ---------------------------------------------------------------------------

def aggregate_pc(selected: list[int], pc_values: np.ndarray) -> float:
    survival = 1.0
    for idx in selected:
        survival *= 1.0 - pc_values[idx]
    return 1.0 - survival


# ---------------------------------------------------------------------------
# Aggregate coverage (mean normalised coverage of selected set)
# ---------------------------------------------------------------------------

def aggregate_coverage(selected: list[int], cov_norm: np.ndarray) -> float:
    if not selected:
        return 0.0
    return float(np.mean(cov_norm[selected]))


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def run_random(
    pc_values: np.ndarray,
    cov_norm: np.ndarray,
    k: int,
    n_runs: int,
    seed: int = 0,
) -> list[dict]:
    rng     = random.Random(seed)
    N       = len(pc_values)
    results = []
    for run_i in range(n_runs):
        selected = rng.sample(range(N), k)
        results.append({
            'run':          run_i,
            'selected':     sorted(selected),
            'aggregate_pc': aggregate_pc(selected, pc_values),
            'mean_cov':     aggregate_coverage(selected, cov_norm),
        })
    return results


# ---------------------------------------------------------------------------
# SA solver (dwave-neal)
# ---------------------------------------------------------------------------

def run_sa(
    Q: np.ndarray,
    pc_values: np.ndarray,
    cov_norm: np.ndarray,
    k: int,
    n_runs: int,
    seed: int = 42,
) -> list[dict]:
    if neal is None or dimod is None:
        print("  WARNING: neal/dimod not installed; SA skipped.")
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
    response = sampler.sample(bqm, num_reads=n_runs, num_sweeps=1000)
    results  = []
    for run_i, sample_rec in enumerate(response.samples()):
        selected = [n for n in range(N) if sample_rec.get(n, 0) == 1]
        results.append({
            'run':          run_i,
            'selected':     sorted(selected),
            'aggregate_pc': aggregate_pc(selected, pc_values),
            'mean_cov':     aggregate_coverage(selected, cov_norm),
        })
    return results


# ---------------------------------------------------------------------------
# SQA solver (dimod SimulatedAnnealingSampler)
# ---------------------------------------------------------------------------

def run_sqa(
    Q: np.ndarray,
    pc_values: np.ndarray,
    cov_norm: np.ndarray,
    k: int,
    n_runs: int,
    seed: int = 123,
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
    response = sampler.sample(bqm, num_reads=n_runs, num_sweeps=1000)
    results  = []
    for run_i, sample_rec in enumerate(response.samples()):
        selected = [n for n in range(N) if sample_rec.get(n, 0) == 1]
        results.append({
            'run':          run_i,
            'selected':     sorted(selected),
            'aggregate_pc': aggregate_pc(selected, pc_values),
            'mean_cov':     aggregate_coverage(selected, cov_norm),
        })
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def stats(results: list[dict]) -> dict:
    if not results:
        return {'best_pc': float('nan'), 'mean_pc': float('nan'),
                'std_pc': float('nan'), 'best_cov': float('nan'), 'all_pc': []}
    pcs  = [r['aggregate_pc'] for r in results]
    covs = [r['mean_cov'] for r in results]
    best_idx = int(np.argmin(pcs))
    return {
        'best_pc':  float(min(pcs)),
        'mean_pc':  float(np.mean(pcs)),
        'std_pc':   float(np.std(pcs)),
        'best_cov': float(covs[best_idx]),
        'all_pc':   pcs,
    }


def oom(pc_ref: float, pc_opt: float) -> str:
    if math.isnan(pc_ref) or math.isnan(pc_opt):
        return "N/A"
    if pc_opt <= 0.0:
        return "inf"
    if pc_ref <= 0.0:
        return "N/A"
    return f"{math.log10(pc_ref / pc_opt):+.2f}"


# ---------------------------------------------------------------------------
# Save best solution
# ---------------------------------------------------------------------------

def save_best(
    results: list[dict],
    path: Path,
    norad_ids: list[int],
    pc_values: np.ndarray,
    cov_norm: np.ndarray,
    raan_values: np.ndarray,
) -> None:
    if not results:
        return
    pcs      = [r['aggregate_pc'] for r in results]
    best_idx = int(np.argmin(pcs))
    selected = results[best_idx]['selected']
    rows     = []
    for idx in selected:
        rows.append({
            'norad_id':     norad_ids[idx],
            'raan_deg':     raan_values[idx],
            'Pc_n':         pc_values[idx],
            'coverage_norm': cov_norm[idx],
        })
    pd.DataFrame(rows).to_csv(path, index=False, float_format='%.6e')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Shell A Bi-Objective QUBO  (Pc + Coverage)")
    print(f"  Objective: min  lambda*(1-Pc_i)*(1-Pc_j) + (1-lambda)*cov_i*cov_j")
    print(f"  Default lambda : {LAMBDA_DEFAULT}   "
          f"(sweep: {LAMBDA_SWEEP})")
    print("=" * 70)

    # --- Check inputs --------------------------------------------------------
    for p in [COVERAGE_CSV]:
        if not p.exists():
            print(f"\n  ERROR: {p.name} not found.")
            print("  Run 'python src/compute_coverage_shell_a.py' first.")
            sys.exit(1)

    # --- Load Pc values ------------------------------------------------------
    if SHELL_A_PC_CSV.exists():
        df_pc_src = pd.read_csv(SHELL_A_PC_CSV)
        print(f"\n  Pc source  : {SHELL_A_PC_CSV.name}  ({len(df_pc_src)} rows)")
        pc_col = 'Pc_n' if 'Pc_n' in df_pc_src.columns else df_pc_src.columns[-1]
    elif MULTISHELL_PC_CSV.exists():
        df_pc_all = pd.read_csv(MULTISHELL_PC_CSV)
        mask      = np.abs(df_pc_all['inc_deg'].values - SHELL_A_INC) <= INC_TOL
        df_pc_src = df_pc_all[mask].reset_index(drop=True)
        print(f"\n  Pc source  : {MULTISHELL_PC_CSV.name} "
              f"(filtered to inc {SHELL_A_INC}+/-{INC_TOL} deg, "
              f"{len(df_pc_src)} rows)")
        pc_col = 'Pc_n'
    else:
        print(f"\n  ERROR: no Pc CSV found.")
        print(f"  Expected: {SHELL_A_PC_CSV.name}  or  {MULTISHELL_PC_CSV.name}")
        sys.exit(1)

    # --- Load coverage -------------------------------------------------------
    df_cov = pd.read_csv(COVERAGE_CSV)
    print(f"  Coverage   : {COVERAGE_CSV.name}  ({len(df_cov)} rows)")

    # --- Merge on raan_deg (tolerance 0.01 deg) ------------------------------
    # Sort both by raan_deg, then align positionally if RAAN grids are identical.
    df_pc_src  = df_pc_src.sort_values('raan_deg').reset_index(drop=True)
    df_cov     = df_cov.sort_values('raan_deg').reset_index(drop=True)

    if len(df_pc_src) != len(df_cov):
        print(f"\n  WARNING: Pc rows ({len(df_pc_src)}) != coverage rows ({len(df_cov)}). "
              f"Using inner merge on raan_deg.")
        df_pc_src['raan_key'] = (df_pc_src['raan_deg'] * 100).round().astype(int)
        df_cov['raan_key']    = (df_cov['raan_deg'] * 100).round().astype(int)
        df_merged = df_pc_src.merge(
            df_cov[['raan_key', 'coverage_raw', 'coverage_norm']],
            on='raan_key', how='inner'
        ).drop(columns='raan_key')
    else:
        raan_diff = np.abs(df_pc_src['raan_deg'].values - df_cov['raan_deg'].values).max()
        if raan_diff > 0.01:
            print(f"\n  WARNING: max RAAN diff = {raan_diff:.4f} deg after sort. "
                  f"Using tolerance merge.")
            df_pc_src['raan_key'] = (df_pc_src['raan_deg'] * 100).round().astype(int)
            df_cov['raan_key']    = (df_cov['raan_deg'] * 100).round().astype(int)
            df_merged = df_pc_src.merge(
                df_cov[['raan_key', 'coverage_raw', 'coverage_norm']],
                on='raan_key', how='inner'
            ).drop(columns='raan_key')
        else:
            df_merged = df_pc_src.copy()
            df_merged['coverage_raw']  = df_cov['coverage_raw'].values
            df_merged['coverage_norm'] = df_cov['coverage_norm'].values

    N = len(df_merged)
    k = N // 2
    P = 10.0 * N

    if k == 0:
        print(f"\n  ERROR: N={N} is too small to select k=N//2 satellites.")
        sys.exit(1)

    pc_values  = df_merged[pc_col].values.astype(np.float64)
    cov_norm   = df_merged['coverage_norm'].values.astype(np.float64)
    norad_ids  = df_merged['norad_id'].tolist()
    raan_vals  = df_merged['raan_deg'].values

    print(f"\n  N          : {N}")
    print(f"  k (select) : {k}  (N//2)")
    print(f"  P (penalty): {P:.0f}")
    print(f"  Pc range   : {pc_values.min():.4e} -- {pc_values.max():.4e}")
    print(f"  Pc mean    : {pc_values.mean():.4e}")
    print(f"  Cov range  : {cov_norm.min():.4f} -- {cov_norm.max():.4f}")
    print(f"  Cov mean   : {cov_norm.mean():.4f}")

    scale_check(pc_values, cov_norm)

    # =========================================================================
    # Main run at LAMBDA_DEFAULT (50 runs each)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"MAIN RUN  lambda = {LAMBDA_DEFAULT}  ({N_RUNS} runs each solver)")
    print(f"{'='*70}")

    Q = build_qubo(pc_values, cov_norm, k, P, LAMBDA_DEFAULT)
    nnz = int((Q != 0).sum())
    print(f"  Q shape : {Q.shape}  |  non-zero : {nnz}")

    print(f"\n--- Random baseline ({N_RUNS} runs) ---")
    rand_results = run_random(pc_values, cov_norm, k, N_RUNS)
    rand_s = stats(rand_results)
    print(f"  Best Pc = {rand_s['best_pc']:.4e}  |  Best cov = {rand_s['best_cov']:.4f}")

    print(f"\n--- SA - neal.SimulatedAnnealingSampler ({N_RUNS} reads) ---")
    sa_results = run_sa(Q, pc_values, cov_norm, k, N_RUNS)
    sa_s = stats(sa_results)
    if sa_results:
        print(f"  Best Pc = {sa_s['best_pc']:.4e}  |  Best cov = {sa_s['best_cov']:.4f}")

    print(f"\n--- SQA - dimod.SimulatedAnnealingSampler ({N_RUNS} reads) ---")
    sqa_results = run_sqa(Q, pc_values, cov_norm, k, N_RUNS)
    sqa_s = stats(sqa_results)
    if sqa_results:
        print(f"  Best Pc = {sqa_s['best_pc']:.4e}  |  Best cov = {sqa_s['best_cov']:.4f}")

    # Results table
    def _fmt(v: float) -> str:
        return f"{v:.4e}" if not math.isnan(v) else "N/A"
    def _fmtf(v: float) -> str:
        return f"{v:.4f}" if not math.isnan(v) else "N/A"

    w = 10
    print()
    print(f"  {'Metric':<26} {'Random':>{w}}  {'SA':>{w}}  {'SQA':>{w}}")
    print(f"  {'-'*26} {'-'*w}  {'-'*w}  {'-'*w}")
    print(f"  {'Best aggregate Pc':<26} {_fmt(rand_s['best_pc']):>{w}}  "
          f"{_fmt(sa_s['best_pc']):>{w}}  {_fmt(sqa_s['best_pc']):>{w}}")
    print(f"  {'Mean aggregate Pc':<26} {_fmt(rand_s['mean_pc']):>{w}}  "
          f"{_fmt(sa_s['mean_pc']):>{w}}  {_fmt(sqa_s['mean_pc']):>{w}}")
    print(f"  {'Std deviation':<26} {_fmt(rand_s['std_pc']):>{w}}  "
          f"{_fmt(sa_s['std_pc']):>{w}}  {_fmt(sqa_s['std_pc']):>{w}}")
    print(f"  {'OOM vs random (best)':<26} {'-':>{w}}  "
          f"{oom(rand_s['best_pc'], sa_s['best_pc']):>{w}}  "
          f"{oom(rand_s['best_pc'], sqa_s['best_pc']):>{w}}")
    print(f"  {'Best-sol mean coverage':<26} {_fmtf(rand_s['best_cov']):>{w}}  "
          f"{_fmtf(sa_s['best_cov']):>{w}}  {_fmtf(sqa_s['best_cov']):>{w}}")

    # Gate
    best_opt = min(
        (sa_s['best_pc']  if sa_results  else float('inf')),
        (sqa_s['best_pc'] if sqa_results else float('inf')),
    )
    if best_opt < rand_s['best_pc']:
        print(f"\n  GATE PASS: SA/SQA best ({best_opt:.4e}) < "
              f"random best ({rand_s['best_pc']:.4e})  OK")
    else:
        print(f"\n  GATE: SA/SQA ({best_opt:.4e}) did not beat "
              f"random ({rand_s['best_pc']:.4e}).")

    # =========================================================================
    # Lambda sweep
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"LAMBDA SWEEP  ({N_SWEEP_RUNS} SA runs each)")
    print(f"{'='*70}")
    sweep_rows = []
    for lv in LAMBDA_SWEEP:
        Q_sw  = build_qubo(pc_values, cov_norm, k, P, lv)
        sa_sw = run_sa(Q_sw, pc_values, cov_norm, k, N_SWEEP_RUNS, seed=int(lv * 1000))
        if not sa_sw:
            continue
        s_sw  = stats(sa_sw)
        print(f"  lambda={lv:.1f} : best_Pc={s_sw['best_pc']:.4e}  "
              f"mean_Pc={s_sw['mean_pc']:.4e}  best_cov={s_sw['best_cov']:.4f}")
        sweep_rows.append({
            'lambda':      lv,
            'best_pc':     s_sw['best_pc'],
            'mean_pc':     s_sw['mean_pc'],
            'std_pc':      s_sw['std_pc'],
            'best_cov':    s_sw['best_cov'],
        })

    # =========================================================================
    # Save outputs
    # =========================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Comparison table
    rows_cmp = []
    for solver, s in [("Random", rand_s), ("SA", sa_s), ("SQA", sqa_s)]:
        rows_cmp.append({
            'solver':           solver,
            'lambda':           LAMBDA_DEFAULT,
            'best_aggregate_pc': s['best_pc'],
            'mean_aggregate_pc': s['mean_pc'],
            'std_aggregate_pc':  s['std_pc'],
            'best_sol_mean_cov': s['best_cov'],
        })
    pd.DataFrame(rows_cmp).to_csv(
        RESULTS_DIR / "coverage_shell_a_comparison.csv",
        index=False, float_format='%.6e'
    )

    # Per-run convergence (used by analyze script)
    if rand_results and sa_results and sqa_results:
        df_runs = pd.DataFrame({
            'run':       list(range(N_RUNS)),
            'random_pc': [r['aggregate_pc'] for r in rand_results[:N_RUNS]],
            'sa_pc':     [r['aggregate_pc'] for r in sa_results[:N_RUNS]],
            'sqa_pc':    [r['aggregate_pc'] for r in sqa_results[:N_RUNS]],
            'random_cov': [r['mean_cov'] for r in rand_results[:N_RUNS]],
            'sa_cov':    [r['mean_cov'] for r in sa_results[:N_RUNS]],
            'sqa_cov':   [r['mean_cov'] for r in sqa_results[:N_RUNS]],
        })
        df_runs.to_csv(
            DATA_DIR / "coverage_shell_a_runs.csv",
            index=False, float_format='%.6e'
        )

    # Best solutions
    save_best(sa_results, RESULTS_DIR / "coverage_shell_a_SA_best.csv",
              norad_ids, pc_values, cov_norm, raan_vals)
    save_best(sqa_results, RESULTS_DIR / "coverage_shell_a_SQA_best.csv",
              norad_ids, pc_values, cov_norm, raan_vals)

    # Lambda sweep CSV
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(
            RESULTS_DIR / "coverage_shell_a_lambda_sweep.csv",
            index=False, float_format='%.6e'
        )

    print(f"\n  Saved results to:")
    print(f"    results/coverage_shell_a_comparison.csv")
    print(f"    results/coverage_shell_a_SA_best.csv")
    print(f"    results/coverage_shell_a_SQA_best.csv")
    print(f"    results/coverage_shell_a_lambda_sweep.csv")
    print(f"    data/coverage_shell_a_runs.csv")
    print("Done.")


if __name__ == "__main__":
    main()
