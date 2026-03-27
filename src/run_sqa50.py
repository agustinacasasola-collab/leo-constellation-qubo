"""
run_sqa50.py
------------
Runs SQA (PathIntegralAnnealingSampler) 50 independent times (seeds 0-49)
on the Arnas Shell 3 QUBO.  Does NOT re-run Random, SA, or Tabu.

Steps
-----
1. Rebuild same QUBO (arnas k=100) — graph + qubo_formulator, no solver change.
2. Run num_reads=50, num_sweeps=200, seed=0 (one vectorised sampler call).
   Each of the 50 reads corresponds to seed index 0-49.
3. For every feasible read compute aggregate Pc; record per-run CSV.
4. Load existing results/method_comparison.csv; update SQA row only.
5. Recompute SA intersect SQA overlap using new SQA best.
6. Regenerate results/spatial_analysis.png (SA data unchanged).
7. Print updated comparison table.

Usage
-----
    python src/run_sqa50.py
    python src/run_sqa50.py --sweeps 100   # faster, ~21 min
    python src/run_sqa50.py --sweeps 1000  # slower, ~206 min
"""

import argparse
import math
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dimod
try:
    from dwave.samplers import PathIntegralAnnealingSampler
except ImportError:
    raise RuntimeError("PathIntegralAnnealingSampler not available. pip install dwave-samplers")

from src.graph_builder import build_graph
from src.qubo_formulator import build_qubo, qubo_to_bqm
from src.classical_annealing import filter_feasible, decode_solution

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')

K             = 100
NUM_READS     = 50
DEFAULT_SWEEPS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def agg_pc(pc_vals) -> float:
    log_surv = sum(math.log1p(-p) for p in pc_vals if p < 1.0)
    return 1.0 - math.exp(log_surv)


def oom(pc_opt: float, pc_rnd: float) -> float:
    return math.log10(pc_rnd / pc_opt) if pc_opt > 0 else float('inf')


# ---------------------------------------------------------------------------
# Step 1 — Rebuild QUBO
# ---------------------------------------------------------------------------

def build_problem():
    print("=" * 65)
    print("STEP 1 — Rebuilding Arnas Shell 3 QUBO  (k=100)")
    print("=" * 65)
    t0 = time.perf_counter()

    cands = pd.read_csv(os.path.join(DATA_DIR, 'arnas_candidates.csv'))
    cands['satellite_id'] = cands['satellite_id'].astype(str)
    print(f"  Loaded {len(cands)} candidates")

    G = build_graph(cands)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    Q, node_idx, P = build_qubo(G, K)
    bqm = qubo_to_bqm(Q)
    print(f"  BQM: {len(bqm.variables)} variables, {len(bqm.quadratic)} interactions")
    print(f"  Penalty P = {P:.0f}")
    print(f"  Elapsed: {time.perf_counter() - t0:.1f}s")

    return bqm, node_idx, cands


# ---------------------------------------------------------------------------
# Step 2 — Run SQA 50 times
# ---------------------------------------------------------------------------

def run_sqa(bqm, node_idx, num_sweeps: int):
    print()
    print("=" * 65)
    print(f"STEP 2 — SQA: {NUM_READS} reads x {num_sweeps} sweeps  (seed=0)")
    print("=" * 65)

    eta_min = 24.7 * (num_sweeps / 100) * NUM_READS / 60
    print(f"  Estimated runtime: ~{eta_min:.0f} min")
    print("  Running ... ", flush=True)

    sampler = PathIntegralAnnealingSampler()
    t0      = time.perf_counter()
    sampleset = sampler.sample(bqm, num_reads=NUM_READS,
                               num_sweeps=num_sweeps, seed=0)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/NUM_READS:.1f}s/read)")

    return sampleset


# ---------------------------------------------------------------------------
# Step 3 — Extract per-run results
# ---------------------------------------------------------------------------

def extract_runs(sampleset, node_idx, cands):
    print()
    print("=" * 65)
    print("STEP 3 — Per-run results")
    print("=" * 65)

    sat_lookup = cands.set_index('satellite_id').to_dict('index')
    idx_to_node = {v: k for k, v in node_idx.items()}

    rows = []
    for run_idx, (sample, energy) in enumerate(
            sampleset.data(['sample', 'energy'])):
        num_sel = sum(int(v) for v in sample.values())
        feasible = (num_sel == K)
        if feasible:
            selected = [idx_to_node[i] for i, v in sample.items() if v == 1]
            pc_vals  = [sat_lookup[s]['pc'] for s in selected]
            run_pc   = agg_pc(pc_vals)
        else:
            selected = []
            run_pc   = float('nan')
        rows.append({
            'seed':       run_idx,
            'feasible':   feasible,
            'num_selected': num_sel,
            'aggregate_pc': run_pc,
            'energy':     energy,
        })

    df = pd.DataFrame(rows)
    n_feas = df['feasible'].sum()
    feas_pcs = df.loc[df['feasible'], 'aggregate_pc']

    print(f"  Feasible runs   : {n_feas}/{NUM_READS}  ({100*n_feas/NUM_READS:.0f}%)")
    if len(feas_pcs):
        print(f"  Aggregate Pc:")
        print(f"    Best  : {feas_pcs.min():.4e}")
        print(f"    Mean  : {feas_pcs.mean():.4e}")
        print(f"    Worst : {feas_pcs.max():.4e}")
        print(f"    Std   : {feas_pcs.std():.4e}")

    runs_path = os.path.join(RESULTS_DIR, 'sqa_runs.csv')
    df.to_csv(runs_path, index=False)
    print(f"  Saved: {runs_path}")

    return df, n_feas


# ---------------------------------------------------------------------------
# Step 3b — Best SQA solution satellites
# ---------------------------------------------------------------------------

def best_sqa_solution(sampleset, node_idx, cands, df_runs):
    sat_lookup  = cands.set_index('satellite_id').to_dict('index')
    idx_to_node = {v: k for k, v in node_idx.items()}

    feas_df = df_runs[df_runs['feasible']].sort_values('aggregate_pc')
    if feas_df.empty:
        raise RuntimeError("No feasible SQA solutions found.")

    best_seed = int(feas_df.iloc[0]['seed'])

    # Re-extract the sample for that seed
    samples = list(sampleset.samples())
    energies = list(sampleset.data_vectors['energy'])
    best_sample = samples[best_seed]
    selected = [idx_to_node[i] for i, v in best_sample.items() if v == 1]
    pc_vals  = [sat_lookup[s]['pc'] for s in selected]
    best_pc  = agg_pc(pc_vals)

    # Save selected_SQA_best.csv
    cands_idx = cands.set_index('satellite_id')
    detail = (
        cands_idx.loc[selected]
        .reset_index()[['satellite_id', 'raan_deg', 'mean_anomaly_deg', 'pc']]
        .rename(columns={'pc': 'Pc_n'})
        .sort_values('raan_deg')
    )
    detail['satellite_id'] = detail['satellite_id'].astype(str)
    path = os.path.join(RESULTS_DIR, 'selected_SQA_best.csv')
    detail.to_csv(path, index=False)
    print(f"\n  Best SQA solution (seed {best_seed}):")
    print(f"    Aggregate Pc : {best_pc:.4e}")
    print(f"    Pc > 0       : {(detail['Pc_n'] > 0).sum()}")
    print(f"    RAAN range   : {detail['raan_deg'].min():.1f} - {detail['raan_deg'].max():.1f} deg")
    print(f"    Saved        : {path}")

    return selected, best_pc


# ---------------------------------------------------------------------------
# Step 4 — Update method_comparison.csv
# ---------------------------------------------------------------------------

def update_comparison(df_runs, best_pc, selected_sqa, n_feas):
    print()
    print("=" * 65)
    print("STEP 4 — Updating method_comparison.csv")
    print("=" * 65)

    # Load existing SA best
    sa_csv    = os.path.join(RESULTS_DIR, 'selected_SA_best.csv')
    sa_detail = pd.read_csv(sa_csv)
    sa_detail['satellite_id'] = sa_detail['satellite_id'].astype(str)
    sa_ids  = set(sa_detail['satellite_id'].tolist())
    sqa_ids = set(str(s) for s in selected_sqa)
    overlap = len(sa_ids & sqa_ids)
    print(f"  SA+SQA overlap : {overlap}/100 satellites (best runs)")

    # Load existing comparison
    comp_path = os.path.join(RESULTS_DIR, 'method_comparison.csv')
    comp = pd.read_csv(comp_path)

    # Load random mean for OOM calc
    rand_row = comp[comp['method'] == 'Random']
    rand_mean = float(rand_row['aggregate_pc'].iloc[0])
    sqa_oom = oom(best_pc, rand_mean)

    # Update SQA row
    mask = comp['method'].str.contains('PathIntegral', na=False)
    comp.loc[mask, 'aggregate_pc']      = best_pc
    comp.loc[mask, 'oom_vs_random']     = round(sqa_oom, 3)
    comp.loc[mask, 'num_reads']         = NUM_READS
    comp.loc[mask, 'feasibility_rate']  = n_feas / NUM_READS
    comp.loc[mask, 'sa_sqa_overlap']    = overlap

    # Also update SA row overlap
    mask_sa = comp['method'].str.contains('Simulated', na=False)
    comp.loc[mask_sa, 'sa_sqa_overlap'] = overlap

    comp.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")

    return comp, rand_mean, sqa_oom, overlap


# ---------------------------------------------------------------------------
# Step 5 — Regenerate spatial_analysis.png
# ---------------------------------------------------------------------------

def regenerate_plot(cands, df_runs, best_pc, selected_sqa):
    print()
    print("=" * 65)
    print("STEP 5 — Regenerating spatial_analysis.png")
    print("=" * 65)

    # Load existing SA / Tabu / random data
    sa_csv   = os.path.join(RESULTS_DIR, 'selected_SA_best.csv')
    rand_csv = os.path.join(RESULTS_DIR, 'random_baseline.csv')
    comp_csv = os.path.join(RESULTS_DIR, 'method_comparison.csv')

    sa_detail = pd.read_csv(sa_csv)
    rand_df   = pd.read_csv(rand_csv, index_col=0)
    comp      = pd.read_csv(comp_csv)

    trials    = [float(rand_df.loc[f'trial_{i:02d}', 'value']) for i in range(30)]
    rand_mean = float(rand_df.loc['random_mean_30trials', 'value'])
    rand_std  = float(rand_df.loc['random_std',           'value'])

    sa_pc    = float(comp.loc[comp['method'].str.contains('Simulated', na=False), 'aggregate_pc'].iloc[0])
    tabu_row = comp[comp['method'].str.contains('Tabu', na=False)]
    tabu_pc  = float(tabu_row['aggregate_pc'].iloc[0]) if not tabu_row.empty else None

    # SQA per-run feasible Pcs
    feas_pcs  = df_runs.loc[df_runs['feasible'], 'aggregate_pc'].values

    cands_idx = cands.set_index('satellite_id')
    sa_detail['satellite_id'] = sa_detail['satellite_id'].astype(str)

    sa_raan  = cands_idx.loc[sa_detail['satellite_id'].tolist(), 'raan_deg'].values
    sqa_ids  = [str(s) for s in selected_sqa]
    sqa_raan = cands_idx.loc[sqa_ids, 'raan_deg'].values

    sa_pc_vals  = sa_detail['Pc_n'].values
    sqa_pc_vals = cands_idx.loc[sqa_ids, 'pc'].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "LEO Constellation Optimization — Arnas Shell 3 | k=100 | 1,656 candidates"
        "\n(SQA: 50 reads x 200 sweeps)",
        fontsize=12, fontweight='bold'
    )

    # ── Subplot 1: Pc distribution ───────────────────────────────────────────
    ax = axes[0]
    all_pc = cands['pc'].values
    nz_all  = all_pc[all_pc > 0]
    pc_min  = max(nz_all.min(), 1e-10)
    bins    = np.logspace(np.log10(pc_min), np.log10(all_pc.max()), 30)

    ax.hist(nz_all, bins=bins, color='gray', alpha=0.40, density=True,
            label=f'All candidates ({len(all_pc):,})')
    if (sa_pc_vals > 0).any():
        ax.hist(sa_pc_vals[sa_pc_vals > 0], bins=bins,
                color='steelblue', alpha=0.75, density=True,
                label=f'SA best  (Pc>0: {(sa_pc_vals>0).sum()})')
    if (sqa_pc_vals > 0).any():
        ax.hist(sqa_pc_vals[sqa_pc_vals > 0], bins=bins,
                color='darkorange', alpha=0.75, density=True,
                label=f'SQA best (Pc>0: {(sqa_pc_vals>0).sum()})')

    ax.set_xscale('log')
    ax.set_xlabel('Individual satellite Pc  (log scale)')
    ax.set_ylabel('Density')
    ax.set_title('Pc Distribution\n(Pc = 0 excluded from plot)')
    ax.legend(fontsize=8)

    # ── Subplot 2: RAAN distribution ─────────────────────────────────────────
    ax = axes[1]
    bins_raan = np.arange(0, 361, 20)
    ax.hist(cands['raan_deg'], bins=bins_raan,
            color='gray', alpha=0.35, density=True,
            label=f'All candidates ({len(cands):,})')
    ax.hist(sa_raan, bins=bins_raan,
            color='steelblue', alpha=0.70, density=True, label='SA best')
    ax.hist(sqa_raan, bins=bins_raan,
            color='darkorange', alpha=0.70, density=True, label='SQA best')
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    ax.set_xlabel('RAAN (deg)')
    ax.set_ylabel('Density')
    ax.set_title('RAAN Distribution\n(orbital plane coverage)')
    ax.legend(fontsize=8)

    # ── Subplot 3: Solution quality per run ───────────────────────────────────
    ax = axes[2]
    rand_arr = np.array(trials)

    ax.axhspan(rand_mean - rand_std, rand_mean + rand_std,
               color='gray', alpha=0.12, label='Random +/- 1 std')
    ax.axhline(rand_mean, color='gray', linestyle='--', linewidth=1.5,
               label=f'Random mean  ({rand_mean:.2e})')
    ax.scatter(np.zeros(len(rand_arr)), rand_arr,
               color='gray', alpha=0.35, s=25, zorder=3)

    # SQA per-run Pc (50 points)
    if len(feas_pcs):
        xs = np.ones(len(feas_pcs)) + np.random.default_rng(42).uniform(
            -0.06, 0.06, len(feas_pcs))
        ax.scatter(xs, feas_pcs, color='darkorange', alpha=0.55, s=30,
                   zorder=4, label=f'SQA runs (n={len(feas_pcs)})')
        ax.scatter([1], [best_pc], color='darkorange', s=160, marker='D',
                   zorder=7, edgecolors='black', linewidths=0.8,
                   label=f'SQA best  ({best_pc:.2e})')

    ax.scatter([2], [sa_pc], color='steelblue', s=160, zorder=7,
               edgecolors='black', linewidths=0.8,
               label=f'SA best   ({sa_pc:.2e})')
    if tabu_pc is not None:
        ax.scatter([2], [tabu_pc], color='seagreen', s=160, marker='s',
                   zorder=7, edgecolors='black', linewidths=0.8,
                   label=f'Tabu best ({tabu_pc:.2e})')

    ax.set_yscale('log')
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Random\n(30 trials)', 'SQA\n(50 runs)', 'Classical\n(SA/Tabu)'])
    ax.set_ylabel('Aggregate constellation Pc  (log scale)')
    ax.set_title('Solution Quality Per Run\n(all 50 SQA runs shown)')
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'spatial_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Step 6 — Print final table
# ---------------------------------------------------------------------------

def print_table(comp, rand_mean, best_pc, sqa_oom, df_runs, overlap):
    print()
    print("=" * 65)
    print("STEP 6 — Updated comparison table")
    print("=" * 65)

    feas_pcs = df_runs.loc[df_runs['feasible'], 'aggregate_pc']
    n_feas   = int(df_runs['feasible'].sum())

    PAPER_RANDOM    = 7.99e-5
    PAPER_OPTIMISED = 4.84e-6
    PAPER_OOM       = math.log10(PAPER_RANDOM / PAPER_OPTIMISED)

    sa_row    = comp[comp['method'].str.contains('Simulated', na=False)].iloc[0]
    tabu_row  = comp[comp['method'].str.contains('Tabu', na=False)]

    print()
    W = 14
    print(f"  {'Method':<36} {'Agg Pc':>{W}} {'OOM vs rnd':>{W}} {'Reads':>{W}} {'Feasible':>{W}}")
    print(f"  {'-'*36} {'-'*W} {'-'*W} {'-'*W} {'-'*W}")
    print(f"  {'Random baseline (30 trials)':<36} {rand_mean:>{W}.4e} {'0.00':>{W}} {'30':>{W}} {'100%':>{W}}")
    print(f"  {'SA (SimulatedAnnealingSampler)':<36} {sa_row['aggregate_pc']:>{W}.4e} {float(sa_row['oom_vs_random']):>{W}.2f} {str(sa_row['num_reads']):>{W}} {'100%':>{W}}")
    if not tabu_row.empty:
        t = tabu_row.iloc[0]
        print(f"  {'Tabu (TabuSampler)':<36} {float(t['aggregate_pc']):>{W}.4e} {float(t['oom_vs_random']):>{W}.2f} {str(t['num_reads']):>{W}} {'100%':>{W}}")
    feas_pct = f"{100*n_feas//NUM_READS}%"
    print(f"  {'SQA (PathIntegralAnnealingSampler)':<36} {best_pc:>{W}.4e} {sqa_oom:>{W}.2f} {str(NUM_READS):>{W}} {feas_pct:>{W}}")
    print()

    if len(feas_pcs):
        print(f"  SQA per-run stats (feasible runs = {n_feas}/{NUM_READS}):")
        print(f"    Best  : {feas_pcs.min():.4e}")
        print(f"    Mean  : {feas_pcs.mean():.4e}")
        print(f"    Worst : {feas_pcs.max():.4e}")
        print(f"    Std   : {feas_pcs.std():.4e}")
    print()
    print(f"  SA + SQA overlap  : {overlap}/100 satellites in best runs")
    print()
    print(f"  Paper target      : {PAPER_OOM:.2f} OOM  ({PAPER_RANDOM:.2e} -> {PAPER_OPTIMISED:.2e})")
    sa_oom_v = float(sa_row['oom_vs_random'])
    verdict  = 'EXCEEDS' if sa_oom_v >= PAPER_OOM else 'BELOW'
    print(f"  SA achieves       : {sa_oom_v:.2f} OOM  -- {verdict} paper target")
    sqa_v    = 'EXCEEDS' if sqa_oom >= PAPER_OOM else 'BELOW'
    print(f"  SQA achieves      : {sqa_oom:.2f} OOM  -- {sqa_v} paper target")
    print()
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SQA 50 times on Arnas QUBO.")
    parser.add_argument('--sweeps', type=int, default=DEFAULT_SWEEPS,
                        help=f'Sweeps per read (default: {DEFAULT_SWEEPS})')
    args = parser.parse_args()

    bqm, node_idx, cands    = build_problem()
    sampleset               = run_sqa(bqm, node_idx, args.sweeps)
    df_runs, n_feas         = extract_runs(sampleset, node_idx, cands)
    selected_sqa, best_pc   = best_sqa_solution(sampleset, node_idx, cands, df_runs)
    comp, rand_mean, sqa_oom, overlap = update_comparison(df_runs, best_pc, selected_sqa, n_feas)
    regenerate_plot(cands, df_runs, best_pc, selected_sqa)
    print_table(comp, rand_mean, best_pc, sqa_oom, df_runs, overlap)


if __name__ == '__main__':
    main()
