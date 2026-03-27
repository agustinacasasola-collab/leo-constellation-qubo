"""
random_baseline.py
------------------
Computes a random-selection baseline for the Arnas Shell 3
constellation problem and compares it against the Table 5 target
from Owens-Fahrner et al. (2025).

Aggregate Pc formula (Eq. 2 of paper — product formula):
    Pc_constellation = 1 - prod_{n=1}^{k} (1 - Pc_n)

Usage
-----
    python src/random_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

INPUT_CSV  = DATA_DIR / "arnas_candidates.csv"
OUTPUT_CSV = RESULTS_DIR / "random_baseline.csv"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
K            = 100       # constellation size (paper Section 4)
N_TRIALS     = 30        # random trials (seeds 0-29)
PAPER_RANDOM = 7.99e-5   # Table 5 random-constellation Pc (Owens-Fahrner 2025)


# ---------------------------------------------------------------------------
# Aggregate Pc  (Eq. 2 — product formula)
# ---------------------------------------------------------------------------

def aggregate_pc(pc_values: np.ndarray) -> float:
    """
    Pc_constellation = 1 - prod(1 - Pc_n)

    Uses log-sum trick for numerical stability when individual Pc values
    span many orders of magnitude (avoids underflow in product).
    """
    log_survival = np.sum(np.log1p(-pc_values))   # sum of log(1 - Pc_n)
    return float(1.0 - np.exp(log_survival))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Random Baseline — Arnas Shell 3 Constellation")
    print(f"  k            = {K}  (constellation size)")
    print(f"  N candidates = 1,656")
    print(f"  Trials       = {N_TRIALS}  (seeds 0-{N_TRIALS - 1})")
    print(f"  Paper target = {PAPER_RANDOM:.2e}  (Table 5 random mean)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    if not INPUT_CSV.exists():
        print(f"\nERROR: {INPUT_CSV} not found.")
        print("Run 'python src/build_arnas_dataset.py' first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    pcs_all = df['pc'].values.astype(float)
    N = len(df)

    # ------------------------------------------------------------------
    # 2. Upfront Pc=0 / Pc>0 breakdown
    # ------------------------------------------------------------------
    n_zero    = int((pcs_all == 0).sum())
    n_nonzero = int((pcs_all > 0).sum())
    pcs_nz    = pcs_all[pcs_all > 0]

    print(f"\nDataset breakdown:")
    print(f"  Total candidates  : {N:,}")
    print(f"  Pc = 0  (safe)    : {n_zero:,}  ({100 * n_zero / N:.1f}%)")
    print(f"  Pc > 0  (at risk) : {n_nonzero:,}  ({100 * n_nonzero / N:.1f}%)")
    print()
    print("  Interpretation: 86.4% of orbital slots have zero computed Pc.")
    print("  A random draw of 100 from 1,656 expects to include")
    expected_nz = K * n_nonzero / N
    print(f"  ~{expected_nz:.1f} non-zero Pc candidates on average.")
    print("  This dilution directly suppresses the random-baseline aggregate Pc.")

    # ------------------------------------------------------------------
    # 3. Random baseline — 30 trials, seeds 0-29
    # ------------------------------------------------------------------
    print(f"\nRunning {N_TRIALS} random trials (k={K}) ...")
    rng = np.random.default_rng()
    trial_results = []

    for seed in range(N_TRIALS):
        rng_s = np.random.default_rng(seed)
        idx   = rng_s.choice(N, size=K, replace=False)
        pc_k  = pcs_all[idx]
        agg   = aggregate_pc(pc_k)
        n_nz  = int((pc_k > 0).sum())
        trial_results.append({
            'seed':        seed,
            'agg_pc':      agg,
            'n_nonzero_k': n_nz,
        })

    trials_df   = pd.DataFrame(trial_results)
    rand_pcs    = trials_df['agg_pc'].values
    rand_mean   = float(rand_pcs.mean())
    rand_std    = float(rand_pcs.std())
    rand_min    = float(rand_pcs.min())
    rand_max    = float(rand_pcs.max())
    rand_nz_avg = float(trials_df['n_nonzero_k'].mean())

    print(f"  Done.  Avg non-zero Pc satellites per trial: {rand_nz_avg:.1f}")

    # ------------------------------------------------------------------
    # 4. Worst-case baseline — top-100 Pc candidates
    # ------------------------------------------------------------------
    top100_idx = np.argsort(pcs_all)[-K:]
    agg_worst  = aggregate_pc(pcs_all[top100_idx])
    pc_top100  = pcs_all[top100_idx]

    print(f"\nWorst-case (top-{K} highest Pc):")
    print(f"  Pc range in selection : [{pc_top100.min():.4e}, {pc_top100.max():.4e}]")
    print(f"  Aggregate Pc          : {agg_worst:.4e}")

    # ------------------------------------------------------------------
    # 5. Best-case baseline — k=100 safest (all Pc=0)
    # ------------------------------------------------------------------
    # If we pick 100 from the Pc=0 pool the aggregate is exactly 0.
    # Compute instead the best-case if we MUST include some non-zero.
    bottom100_idx = np.argsort(pcs_all)[:K]
    agg_best      = aggregate_pc(pcs_all[bottom100_idx])

    # ------------------------------------------------------------------
    # 6. Non-zero-pool-only random baseline (diagnostic)
    # ------------------------------------------------------------------
    print(f"\nDiagnostic: random baseline selecting ONLY from {n_nonzero}"
          f" non-zero Pc candidates ...")
    nz_idx = np.where(pcs_all > 0)[0]
    nz_K   = min(K, n_nonzero)
    nz_trials = []
    for seed in range(N_TRIALS):
        rng_s = np.random.default_rng(seed + 1000)
        idx_s = rng_s.choice(nz_idx, size=nz_K, replace=False)
        nz_trials.append(aggregate_pc(pcs_all[idx_s]))
    nz_mean = float(np.mean(nz_trials))
    nz_std  = float(np.std(nz_trials))

    # ------------------------------------------------------------------
    # 7. Comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("COMPARISON TABLE")
    print("=" * 65)
    print(f"  {'Metric':<36} {'Your result':>12}  {'Paper':>10}")
    print(f"  {'-'*36} {'-'*12}  {'-'*10}")
    print(f"  {'Random mean (30 trials, k=100)':<36} {rand_mean:>12.4e}  {PAPER_RANDOM:>10.2e}")
    print(f"  {'Random std':<36} {rand_std:>12.4e}  {'—':>10}")
    print(f"  {'Random min / max':<36} {rand_min:.2e} / {rand_max:.2e}  {'—':>10}")
    print(f"  {'Worst case (top-100 Pc)':<36} {agg_worst:>12.4e}  {'—':>10}")
    print(f"  {'Best case (bottom-100 Pc)':<36} {agg_best:>12.4e}  {'—':>10}")
    print(f"  {'Non-zero pool only — mean':<36} {nz_mean:>12.4e}  {'—':>10}")
    print(f"  {'Non-zero pool only — std':<36} {nz_std:>12.4e}  {'—':>10}")

    # ------------------------------------------------------------------
    # 8. Gap diagnosis
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("GAP DIAGNOSIS")
    print("=" * 65)

    ratio = nz_mean / PAPER_RANDOM if PAPER_RANDOM > 0 else float('inf')

    if rand_mean < PAPER_RANDOM * 0.1:
        print(f"  random mean ({rand_mean:.2e}) << paper ({PAPER_RANDOM:.2e})")
        print(f"  PRIMARY CAUSE: {n_zero:,} / {N:,} ({100*n_zero/N:.1f}%) candidates")
        print(f"  have Pc = 0.  A random draw of {K} expects only")
        print(f"  ~{expected_nz:.1f} non-zero Pc satellites, driving")
        print(f"  aggregate Pc down by ~x{K/max(expected_nz,0.01):.0f}.")
        print()
        print(f"  Non-zero pool mean : {nz_mean:.4e}  (x{ratio:.2f} vs paper)")
        if 0.5 <= ratio <= 2.0:
            print(f"  -> The Pc formula is ORDER-OF-MAGNITUDE correct.")
            print(f"     The gap vs paper is explained by Pc=0 dilution.")
        elif ratio < 0.5:
            print(f"  -> Non-zero pool is LOWER than paper by {1/ratio:.1f}x.")
            print(f"     Possible causes: screening misses some conjunctions;")
            print(f"     catalog epoch too old; TLE propagation uncertainty.")
        else:
            print(f"  -> Non-zero pool is HIGHER than paper by {ratio:.1f}x.")
            print(f"     Possible causes: sigma/rho values; catalog size differs.")
    elif rand_mean > PAPER_RANDOM * 10:
        print(f"  random mean ({rand_mean:.2e}) >> paper ({PAPER_RANDOM:.2e})")
        print("  Possible causes: sigma too small; catalog larger than paper's.")
    else:
        print(f"  random mean ({rand_mean:.2e}) is within 10x of paper ({PAPER_RANDOM:.2e}).")
        print(f"  Ratio: {rand_mean / PAPER_RANDOM:.2f}x — reasonable agreement.")

    # ------------------------------------------------------------------
    # 9. Save CSV
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {'metric': 'random_mean_30trials', 'value': rand_mean,  'paper': PAPER_RANDOM},
        {'metric': 'random_std',           'value': rand_std,   'paper': None},
        {'metric': 'random_min',           'value': rand_min,   'paper': None},
        {'metric': 'random_max',           'value': rand_max,   'paper': None},
        {'metric': 'worst_case_top100',    'value': agg_worst,  'paper': None},
        {'metric': 'best_case_bottom100',  'value': agg_best,   'paper': None},
        {'metric': 'nz_pool_mean',         'value': nz_mean,    'paper': None},
        {'metric': 'nz_pool_std',          'value': nz_std,     'paper': None},
    ]
    df_summary = pd.DataFrame(summary_rows)

    # Append per-trial detail
    trials_df['metric'] = trials_df['seed'].apply(lambda s: f'trial_{s:02d}')
    trials_df = trials_df.rename(columns={'agg_pc': 'value'})
    trials_df['paper'] = None
    df_all = pd.concat([df_summary,
                        trials_df[['metric', 'value', 'paper']]],
                       ignore_index=True)
    df_all.to_csv(OUTPUT_CSV, index=False, float_format='%.6e')
    print(f"\n  Results saved to: {OUTPUT_CSV}")
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
