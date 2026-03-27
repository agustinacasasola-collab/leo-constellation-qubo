"""
analyze_results.py
------------------
Loads saved results and produces a clean comparison report.
Does NOT re-run any solvers.

Data sources
------------
  results/solutions_arnas.csv   — best solution per solver (SA, Tabu, SQA)
  results/random_baseline.csv   — 30 random trial Pc values + stats
  data/arnas_candidates.csv     — full candidate pool (raan, mean_anomaly, pc)

Usage
-----
    python src/analyze_results.py

Outputs
-------
  results/spatial_analysis.png
  results/selected_SA_best.csv
  results/selected_SQA_best.csv
  results/method_comparison.csv
"""

import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA_DIR    = os.path.join(ROOT, 'data')

# Pipeline run statistics logged from:
#   python main.py --data arnas --k 100 --num-reads 50
# Samplesets are not serialised to disk; per-run stats are captured here.
_RUN_STATS = {
    'simulated_annealing': {'num_reads': 50, 'num_feasible': 50, 'feasibility_rate': 1.0},
    'tabu_search':         {'num_reads': 50, 'num_feasible': 50, 'feasibility_rate': 1.0},
    'sqa_path_integral':   {'num_reads':  5, 'num_feasible':  5, 'feasibility_rate': 1.0},
}

PAPER_RANDOM     = 7.99e-5   # Table 5, Shell 3 random mean
PAPER_OPTIMISED  = 4.84e-6   # Table 5, Shell 3 optimised (SA/QA best)
PAPER_OOM        = math.log10(PAPER_RANDOM / PAPER_OPTIMISED)  # ~1.22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def agg_pc(pc_vals) -> float:
    """Aggregate Pc: 1 − exp(Σ log(1−Pc_n))."""
    log_surv = sum(math.log1p(-p) for p in pc_vals if p < 1.0)
    return 1.0 - math.exp(log_surv)


def oom(pc_opt: float, pc_rnd: float) -> float:
    """Orders-of-magnitude reduction: log10(pc_rnd / pc_opt)."""
    if pc_opt <= 0:
        return float('inf')
    return math.log10(pc_rnd / pc_opt)


# ---------------------------------------------------------------------------
# STEP 1 — Load files
# ---------------------------------------------------------------------------

def step1_load():
    print("=" * 65)
    print("STEP 1 — Finding and loading existing results")
    print("=" * 65)

    # solutions_arnas.csv
    sols_path = os.path.join(RESULTS_DIR, 'solutions_arnas.csv')
    sols = pd.read_csv(sols_path)
    sols['satellite_id'] = sols['satellite_id'].astype(str)
    print(f"\n  solutions_arnas.csv")
    print(f"    Path    : {sols_path}")
    print(f"    Rows    : {len(sols)}")
    print(f"    Columns : {list(sols.columns)}")
    per = sols.groupby('solver').size().to_dict()
    print(f"    Per solver: {per}")

    # random_baseline.csv
    rand_path = os.path.join(RESULTS_DIR, 'random_baseline.csv')
    rand_df   = pd.read_csv(rand_path, index_col=0)
    trials    = [float(rand_df.loc[f'trial_{i:02d}', 'value']) for i in range(30)]
    rand_mean = float(rand_df.loc['random_mean_30trials', 'value'])
    rand_std  = float(rand_df.loc['random_std',           'value'])
    print(f"\n  random_baseline.csv")
    print(f"    Path    : {rand_path}")
    print(f"    Trials  : 30  (trial_00 … trial_29)")
    print(f"    Mean Pc : {rand_mean:.4e}")
    print(f"    Std  Pc : {rand_std:.4e}")
    print(f"    Min  Pc : {min(trials):.4e}")
    print(f"    Max  Pc : {max(trials):.4e}")

    # arnas_candidates.csv
    cands_path = os.path.join(DATA_DIR, 'arnas_candidates.csv')
    cands = pd.read_csv(cands_path)
    cands['satellite_id'] = cands['satellite_id'].astype(str)
    print(f"\n  arnas_candidates.csv")
    print(f"    Path    : {cands_path}")
    print(f"    Rows    : {len(cands)}")
    print(f"    Columns : {list(cands.columns)}")

    return sols, trials, rand_mean, rand_std, cands


# ---------------------------------------------------------------------------
# STEP 2 — Performance comparison table
# ---------------------------------------------------------------------------

def step2_performance(sols, trials, rand_mean, rand_std):
    print()
    print("=" * 65)
    print("STEP 2 — Performance comparison table")
    print("=" * 65)

    sa_sats   = sols[sols['solver'] == 'simulated_annealing']
    tabu_sats = sols[sols['solver'] == 'tabu_search']
    sqa_sats  = sols[sols['solver'] == 'sqa_path_integral']

    pc_sa   = agg_pc(sa_sats['pc'].tolist())
    pc_tabu = agg_pc(tabu_sats['pc'].tolist())
    pc_sqa  = agg_pc(sqa_sats['pc'].tolist())

    rand_arr   = np.array(trials)
    rand_best  = float(rand_arr.min())
    rand_worst = float(rand_arr.max())
    rand_stdv  = float(rand_arr.std())

    sa_oom   = oom(pc_sa,   rand_mean)
    tabu_oom = oom(pc_tabu, rand_mean)
    sqa_oom  = oom(pc_sqa,  rand_mean)

    rs = _RUN_STATS

    W = 14
    print()
    print(f"  {'Metric':<28} {'Random':>{W}} {'SA':>{W}} {'Tabu':>{W}} {'SQA':>{W}}")
    print(f"  {'-'*28} {'-'*W} {'-'*W} {'-'*W} {'-'*W}")
    print(f"  {'Mean aggregate Pc':<28} {rand_mean:>{W}.4e} {'(best run)':>{W}} {'(best run)':>{W}} {'(best run)':>{W}}")
    print(f"  {'Best aggregate Pc':<28} {rand_best:>{W}.4e} {pc_sa:>{W}.4e} {pc_tabu:>{W}.4e} {pc_sqa:>{W}.4e}")
    print(f"  {'Worst aggregate Pc':<28} {rand_worst:>{W}.4e} {'N/A':>{W}} {'N/A':>{W}} {'N/A':>{W}}")
    print(f"  {'Std deviation':<28} {rand_stdv:>{W}.4e} {'N/A':>{W}} {'N/A':>{W}} {'N/A':>{W}}")
    print(f"  {'Reduction vs random':<28} {'—':>{W}} {sa_oom:>{W-1}.2f}x {tabu_oom:>{W-1}.2f}x {sqa_oom:>{W-1}.2f}x")
    sa_s   = f'{sa_oom:.2f} OOM'
    tabu_s = f'{tabu_oom:.2f} OOM'
    sqa_s  = f'{sqa_oom:.2f} OOM'
    print(f"  {'  (orders of magnitude)':<28} {'—':>{W}} {sa_s:>{W}} {tabu_s:>{W}} {sqa_s:>{W}}")
    fr_sa   = f"{rs['simulated_annealing']['feasibility_rate']:.0%}"
    fr_tabu = f"{rs['tabu_search']['feasibility_rate']:.0%}"
    fr_sqa  = f"{rs['sqa_path_integral']['feasibility_rate']:.0%}"
    print(f"  {'Feasible runs (%)':<28} {'100%':>{W}} {fr_sa:>{W}} {fr_tabu:>{W}} {fr_sqa:>{W}}")
    print()
    print(f"  Paper target  (Shell 3, 2024/25 catalog):")
    print(f"    random={PAPER_RANDOM:.2e}  ->  optimised={PAPER_OPTIMISED:.2e}  ({PAPER_OOM:.2f} OOM)")

    return pc_sa, pc_tabu, pc_sqa, sa_oom, tabu_oom, sqa_oom, sa_sats, tabu_sats, sqa_sats


# ---------------------------------------------------------------------------
# STEP 3 — Consistency analysis
# ---------------------------------------------------------------------------

def step3_consistency(sa_sats, sqa_sats):
    print()
    print("=" * 65)
    print("STEP 3 — Consistency analysis")
    print("=" * 65)
    print()
    print("  NOTE: solutions_arnas.csv stores the BEST solution per solver.")
    print("  Samplesets (all reads) were not serialised to disk. Per-run")
    print("  satellite overlap and Pc variance are therefore not computable")
    print("  from saved files alone. Statistics below use the logged run")
    print("  (main.py --data arnas --k 100 --num-reads 50).")
    print()

    sa_ids  = set(sa_sats['satellite_id'])
    sqa_ids = set(sqa_sats['satellite_id'])
    overlap = len(sa_ids & sqa_ids)

    rs = _RUN_STATS
    sa_r  = rs['simulated_annealing']
    sqa_r = rs['sqa_path_integral']

    W = 18
    print(f"  {'Consistency metric':<28} {'SA':>{W}} {'SQA':>{W}}")
    print(f"  {'-'*28} {'-'*W} {'-'*W}")
    print(f"  {'Runs performed':<28} {str(sa_r['num_reads']):>{W}} {str(sqa_r['num_reads']):>{W}}")
    sa_feas  = f"{sa_r['num_feasible']} (100%)"
    sqa_feas = f"{sqa_r['num_feasible']} (100%)"
    print(f"  {'Feasible runs':<28} {sa_feas:>{W}} {sqa_feas:>{W}}")
    print(f"  {'Best energy':<28} {sa_sats['energy'].iloc[0]:>{W}.2f} {sqa_sats['energy'].iloc[0]:>{W}.2f}")
    print(f"  {'Core (all runs)':<28} {'not available':>{W}} {'not available':>{W}}")
    print(f"  {'Stable (>=20 runs)':<28} {'not available':>{W}} {'not available':>{W}}")
    print(f"  {'SA+SQA (best runs)':<28} {f'{overlap}/100':>{W}} {f'{overlap}/100':>{W}}")
    print()
    print("  To enable full per-run overlap statistics, modify solve_simulated_annealing")
    print("  and solve_sqa to save samplesets to disk (e.g. dimod.SampleSet.to_serializable).")

    return overlap


# ---------------------------------------------------------------------------
# STEP 4 — Spatial analysis plot
# ---------------------------------------------------------------------------

def step4_plot(sols, cands, pc_sa, pc_tabu, pc_sqa, rand_mean, rand_std, trials,
               sa_sats, sqa_sats, tabu_sats):
    print()
    print("=" * 65)
    print("STEP 4 — Generating spatial analysis plot")
    print("=" * 65)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "LEO Constellation Optimization — Arnas Shell 3 | k=100 | 1,656 candidates",
        fontsize=13, fontweight='bold'
    )

    cands_idx = cands.set_index('satellite_id')

    # ── Subplot 1: Pc distribution (log scale) ──────────────────────────────
    ax = axes[0]
    all_pc  = cands['pc'].values
    sa_pc   = sa_sats['pc'].values
    sqa_pc  = sqa_sats['pc'].values

    nonzero_all = all_pc[all_pc > 0]
    pc_min = max(nonzero_all.min(), 1e-10)
    pc_max = all_pc.max()
    bins = np.logspace(np.log10(pc_min), np.log10(pc_max), 30)

    ax.hist(nonzero_all, bins=bins,
            color='gray', alpha=0.45, density=True,
            label=f'All candidates ({len(all_pc):,})')
    if (sa_pc > 0).any():
        ax.hist(sa_pc[sa_pc > 0], bins=bins,
                color='steelblue', alpha=0.75, density=True,
                label=f'SA best  (Pc>0: {(sa_pc>0).sum()})')
    if (sqa_pc > 0).any():
        ax.hist(sqa_pc[sqa_pc > 0], bins=bins,
                color='darkorange', alpha=0.75, density=True,
                label=f'SQA best (Pc>0: {(sqa_pc>0).sum()})')

    ax.set_xscale('log')
    ax.set_xlabel('Individual satellite Pc  (log scale)')
    ax.set_ylabel('Density')
    ax.set_title('Pc Distribution\n(Pc = 0 satellites excluded)')
    ax.legend(fontsize=8)

    # ── Subplot 2: RAAN distribution ────────────────────────────────────────
    ax = axes[1]
    sa_raan  = cands_idx.loc[sa_sats['satellite_id'].tolist(),  'raan_deg'].values
    sqa_raan = cands_idx.loc[sqa_sats['satellite_id'].tolist(), 'raan_deg'].values

    bins_raan = np.arange(0, 361, 20)
    ax.hist(cands['raan_deg'], bins=bins_raan,
            color='gray', alpha=0.40, density=True,
            label=f'All candidates ({len(cands):,})')
    ax.hist(sa_raan, bins=bins_raan,
            color='steelblue', alpha=0.75, density=True, label='SA best')
    ax.hist(sqa_raan, bins=bins_raan,
            color='darkorange', alpha=0.75, density=True, label='SQA best')
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    ax.set_xlabel('RAAN (deg)')
    ax.set_ylabel('Density')
    ax.set_title('RAAN Distribution\n(orbital plane coverage)')
    ax.legend(fontsize=8)

    # ── Subplot 3: Solution quality (log Pc) ────────────────────────────────
    ax = axes[2]
    rand_arr = np.array(trials)

    ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
               color='gray', alpha=0.15, label='Random +/- 1 std')
    ax.axhline(rand_mean, color='gray', linestyle='--', linewidth=1.5,
               label=f'Random mean  ({rand_mean:.2e})')

    # Individual random trials as scatter
    ax.scatter(np.zeros(len(rand_arr)), rand_arr,
               color='gray', alpha=0.4, s=30, zorder=3)

    # Solver best solutions
    ax.scatter([1], [pc_sa],   color='steelblue',  s=140, zorder=6,
               label=f'SA best   ({pc_sa:.2e})')
    ax.scatter([1], [pc_tabu], color='seagreen',    s=140, marker='s', zorder=6,
               label=f'Tabu best ({pc_tabu:.2e})')
    ax.scatter([1], [pc_sqa],  color='darkorange',  s=140, marker='D', zorder=6,
               label=f'SQA best  ({pc_sqa:.2e})')

    ax.set_yscale('log')
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Random\n(30 trials)', 'Optimised\n(best run)'])
    ax.set_ylabel('Aggregate constellation Pc  (log scale)')
    ax.set_title('Solution Quality\n(best run per solver vs random baseline)')
    ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'spatial_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# STEP 5 — Selected satellite CSVs
# ---------------------------------------------------------------------------

def step5_selected(sa_sats, sqa_sats, cands):
    print()
    print("=" * 65)
    print("STEP 5 — Selected satellite details")
    print("=" * 65)

    cands_idx = cands.set_index('satellite_id')

    def save_selected(sats_df, label):
        ids = sats_df['satellite_id'].tolist()
        detail = (
            cands_idx.loc[ids]
            .reset_index()[['satellite_id', 'raan_deg', 'mean_anomaly_deg', 'pc']]
            .rename(columns={'pc': 'Pc_n'})
            .sort_values('raan_deg')
        )
        detail['satellite_id'] = detail['satellite_id'].astype(str)
        path = os.path.join(RESULTS_DIR, f'selected_{label}_best.csv')
        detail.to_csv(path, index=False)
        print(f"\n  Saved: {path}  ({len(detail)} satellites)")
        print(f"    RAAN range  : {detail['raan_deg'].min():.1f} - {detail['raan_deg'].max():.1f} deg")
        print(f"    Pc > 0      : {(detail['Pc_n'] > 0).sum()}")
        print(f"    Best Pc_n   : {detail['Pc_n'].max():.3e}")
        return set(ids)

    sa_ids  = save_selected(sa_sats,  'SA')
    sqa_ids = save_selected(sqa_sats, 'SQA')

    overlap = len(sa_ids & sqa_ids)
    print(f"\n  SA and SQA agree on {overlap}/100 satellites in best run")
    return overlap


# ---------------------------------------------------------------------------
# STEP 6 — Final summary + method_comparison.csv
# ---------------------------------------------------------------------------

def step6_summary(rand_mean, rand_std, pc_sa, pc_tabu, pc_sqa,
                  sa_oom, tabu_oom, sqa_oom, overlap):
    print()
    print("=" * 65)
    print("STEP 6 — Final summary")
    print("=" * 65)
    print()

    best_method = min(
        [('SA', pc_sa, sa_oom), ('Tabu', pc_tabu, tabu_oom), ('SQA', pc_sqa, sqa_oom)],
        key=lambda t: t[1]
    )
    best_consist = 'SA'   # most reads (50), well-sampled

    print(f"  Random mean Pc    : {rand_mean:.4e}")
    print(f"  SA best Pc        : {pc_sa:.4e}  ({sa_oom:.2f} OOM below random)")
    print(f"  SQA best Pc       : {pc_sqa:.4e}  ({sqa_oom:.2f} OOM below random)")
    print(f"  Best method by Pc : {best_method[0]}  ({best_method[1]:.4e},  {best_method[2]:.2f} OOM)")
    print(f"  Best consistency  : {best_consist}  (50 reads, 100% feasible)")
    print(f"  SA+SQA overlap    : {overlap}/100 satellites in best runs")
    print()
    print(f"  Paper target      : {PAPER_OOM:.2f} OOM reduction  ({PAPER_RANDOM:.2e} -> {PAPER_OPTIMISED:.2e})")
    verdict = 'EXCEEDS' if sa_oom >= PAPER_OOM else 'BELOW'
    print(f"  SA achieves       : {sa_oom:.2f} OOM  — {verdict} paper target")
    print(f"  Note: absolute Pc higher due to catalog growth (2026 vs 2024/25);")
    print(f"        reduction ratio is the fair cross-epoch comparison metric.")

    # Save method_comparison.csv
    rs = _RUN_STATS
    rows = [
        {'method': 'Random',
         'aggregate_pc': rand_mean,
         'oom_vs_random': 0.0,
         'num_reads': 30,
         'feasibility_rate': 1.0,
         'sa_sqa_overlap': '—'},
        {'method': 'SA (SimulatedAnnealingSampler)',
         'aggregate_pc': pc_sa,
         'oom_vs_random': round(sa_oom, 3),
         'num_reads': rs['simulated_annealing']['num_reads'],
         'feasibility_rate': rs['simulated_annealing']['feasibility_rate'],
         'sa_sqa_overlap': overlap},
        {'method': 'Tabu (TabuSampler)',
         'aggregate_pc': pc_tabu,
         'oom_vs_random': round(tabu_oom, 3),
         'num_reads': rs['tabu_search']['num_reads'],
         'feasibility_rate': rs['tabu_search']['feasibility_rate'],
         'sa_sqa_overlap': '—'},
        {'method': 'SQA (PathIntegralAnnealingSampler)',
         'aggregate_pc': pc_sqa,
         'oom_vs_random': round(sqa_oom, 3),
         'num_reads': rs['sqa_path_integral']['num_reads'],
         'feasibility_rate': rs['sqa_path_integral']['feasibility_rate'],
         'sa_sqa_overlap': overlap},
        {'method': 'Paper SA/QA (Owens-Fahrner 2025)',
         'aggregate_pc': PAPER_OPTIMISED,
         'oom_vs_random': round(PAPER_OOM, 3),
         'num_reads': '—',
         'feasibility_rate': '—',
         'sa_sqa_overlap': '—'},
    ]
    comp_path = os.path.join(RESULTS_DIR, 'method_comparison.csv')
    pd.DataFrame(rows).to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")
    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sols, trials, rand_mean, rand_std, cands = step1_load()
    (pc_sa, pc_tabu, pc_sqa,
     sa_oom, tabu_oom, sqa_oom,
     sa_sats, tabu_sats, sqa_sats) = step2_performance(sols, trials, rand_mean, rand_std)
    step3_consistency(sa_sats, sqa_sats)
    step4_plot(sols, cands, pc_sa, pc_tabu, pc_sqa,
               rand_mean, rand_std, trials,
               sa_sats, sqa_sats, tabu_sats)
    overlap = step5_selected(sa_sats, sqa_sats, cands)
    step6_summary(rand_mean, rand_std, pc_sa, pc_tabu, pc_sqa,
                  sa_oom, tabu_oom, sqa_oom, overlap)


if __name__ == '__main__':
    main()
