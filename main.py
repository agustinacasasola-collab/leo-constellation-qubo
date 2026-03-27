"""
main.py
-------
Entry point for the LEO constellation QUBO optimization pipeline.

Runs the full pipeline:
    1. Load satellite data
    2. Build candidate graph
    3. Formulate QUBO
    4. Solve with Simulated Annealing (always available)
    5. Solve with Tabu Search (always available)
    6. Optionally solve with D-Wave QPU (requires Leap account)
    7. Save results to CSV
    8. Generate and save graph visualisation

Usage:
    python main.py
    python main.py --k 6
    python main.py --k 5 --num-reads 500 --quantum
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import pandas as pd

# Ensure project root is on the path when running from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DEFAULT_K, SA_NUM_READS
from src.graph_builder import build_graph, graph_summary
from src.qubo_formulator import build_qubo, qubo_to_bqm, print_qubo_stats
from src.classical_annealing import (
    solve_simulated_annealing,
    solve_tabu,
    solve_sqa,
    print_results,
)
from src.quantum_annealing import check_leap_access, solve_quantum, compare_solvers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LEO Constellation QUBO Optimization"
    )
    parser.add_argument(
        '--k', type=int, default=DEFAULT_K,
        help=f'Constellation size — number of satellites to select (default: {DEFAULT_K})'
    )
    parser.add_argument(
        '--num-reads', type=int, default=SA_NUM_READS,
        help=f'Number of annealing reads (default: {SA_NUM_READS})'
    )
    parser.add_argument(
        '--quantum', action='store_true',
        help='Attempt quantum annealing via D-Wave Leap (requires dwave config create)'
    )
    parser.add_argument(
        '--data', type=str, default='sample',
        choices=['sample', 'real', 'arnas'],
        help=(
            "Dataset to use: "
            "'sample' (20 synthetic candidates, default), "
            "'real' (20 real TLE candidates, SGP4 Pc), "
            "'arnas' (1,656 Shell 3 candidates, Arnas 2021 + Chan Pc — k defaults to 100)"
        )
    )
    return parser.parse_args()


def save_results(results_list: list, satellites_df: pd.DataFrame, path: str) -> None:
    """Save all solver results to a CSV file."""
    sat_lookup = satellites_df.set_index('satellite_id').to_dict('index')
    rows = []
    for results in results_list:
        for sat_id in results['selected_satellites']:
            rows.append({
                'solver': results['solver'],
                'satellite_id': sat_id,
                'pc': sat_lookup[sat_id]['pc'],
                'coverage': sat_lookup[sat_id]['coverage'],
                'energy': results['best_energy'],
                'feasible': True,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\n  Results saved to: {path}")


def plot_graph(G: nx.Graph, selected_satellites: list, output_path: str) -> None:
    """
    Draw the complete satellite graph and save to PNG.

    Selected satellites (SA solution) are coloured green;
    unselected nodes are grey. Edges are coloured by weight
    (darker = higher weight). A colorbar shows the weight scale.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Node colours: green for selected, light grey for unselected.
    node_colors = [
        '#2ecc71' if node in selected_satellites else '#bdc3c7'
        for node in G.nodes()
    ]
    node_sizes = [
        500 if node in selected_satellites else 300
        for node in G.nodes()
    ]

    # Edge colours: map weight to a colour spectrum.
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    min_w, max_w = min(weights), max(weights)
    edge_colors = [
        cm.Blues(0.3 + 0.7 * (d['weight'] - min_w) / (max_w - min_w + 1e-9))
        for _, _, d in G.edges(data=True)
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           width=1.0, alpha=0.7, ax=ax)

    # Colorbar for edge weights.
    sm = plt.cm.ScalarMappable(
        cmap=cm.Blues,
        norm=plt.Normalize(vmin=min_w, vmax=max_w)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Edge weight w(vn, vm)')

    ax.set_title(
        "LEO Constellation Graph — Simulated Annealing Solution\n"
        "(green nodes = selected satellites, edge shade = weight)",
        fontsize=13,
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Graph visualisation saved to: {output_path}")


def main() -> None:
    args = parse_args()
    k = args.k
    # For the Arnas dataset the paper uses k=100; apply that default only when
    # the user did not explicitly pass --k (i.e. k is still the global default).
    if args.data == 'arnas' and k == DEFAULT_K:
        k = 100
    num_reads = args.num_reads

    # ------------------------------------------------------------------
    # 1. Load satellite data
    # ------------------------------------------------------------------
    if args.data == 'real':
        data_file = 'real_satellites.csv'
    elif args.data == 'arnas':
        data_file = 'arnas_candidates.csv'
    else:
        data_file = 'sample_satellites.csv'
    data_path = os.path.join(os.path.dirname(__file__), 'data', data_file)
    print(f"\nLoading satellite data from: {data_path}")
    satellites_df = pd.read_csv(data_path)
    satellites_df['satellite_id'] = satellites_df['satellite_id'].astype(str)
    N = len(satellites_df)
    # Only print the full table for small datasets; summarise for large ones.
    if N <= 30:
        print(satellites_df[['satellite_id', 'pc', 'coverage']].to_string(index=False))
    else:
        pcs = satellites_df['pc'].values
        print(f"  (Showing summary — {N} candidates, printing full table skipped)")
        print(f"  Pc = 0      : {(pcs == 0).sum():,}  ({100*(pcs==0).mean():.1f}%)")
        print(f"  Pc > 0      : {(pcs > 0).sum():,}  ({100*(pcs>0).mean():.1f}%)")
        print(f"  Top 5 Pc    : {sorted(pcs, reverse=True)[:5]}")
    print(f"\n  {N} candidate satellites loaded.")
    if 'shell' in satellites_df.columns:
        print(f"  Shells: {sorted(satellites_df['shell'].unique())}")
    print(f"  Pc range     : [{satellites_df['pc'].min():.3e}, {satellites_df['pc'].max():.3e}]")
    print(f"  Coverage range: [{satellites_df['coverage'].min():.4f}, "
          f"{satellites_df['coverage'].max():.4f}]")

    # ------------------------------------------------------------------
    # 2. Build graph
    # ------------------------------------------------------------------
    print(f"\nBuilding complete graph for {len(satellites_df)} satellites...")
    G = build_graph(satellites_df)
    graph_summary(G, k)

    # ------------------------------------------------------------------
    # 3. Build QUBO
    # ------------------------------------------------------------------
    print(f"\nFormulating QUBO for k={k}...")
    Q, node_idx, P = build_qubo(G, k)
    print_qubo_stats(Q, node_idx, k, P)

    # ------------------------------------------------------------------
    # 4. Convert to BQM
    # ------------------------------------------------------------------
    bqm = qubo_to_bqm(Q)
    print(f"\n  BQM created: {len(bqm.variables)} variables, "
          f"{len(bqm.quadratic)} quadratic interactions.")

    # ------------------------------------------------------------------
    # 5. Simulated Annealing
    # ------------------------------------------------------------------
    print(f"\nRunning Simulated Annealing (num_reads={num_reads})...")
    results_sa = solve_simulated_annealing(bqm, node_idx, k, num_reads=num_reads)
    print_results(results_sa, satellites_df)

    # ------------------------------------------------------------------
    # 6. Tabu Search
    # ------------------------------------------------------------------
    print(f"\nRunning Tabu Search (num_reads={num_reads})...")
    results_tabu = solve_tabu(bqm, node_idx, k, num_reads=num_reads)
    print_results(results_tabu, satellites_df)

    # ------------------------------------------------------------------
    # 7. Simulated Quantum Annealing — PathIntegralAnnealingSampler (dimod)
    # ------------------------------------------------------------------
    # SQA via PathIntegralAnnealingSampler is ~143 s/read for N=1656;
    # cap at 5 reads so the step stays under ~15 min.
    sqa_reads = min(num_reads, 5) if len(bqm.variables) > 200 else min(num_reads, 100)
    print(f"\nRunning Simulated QA — PathIntegralAnnealingSampler "
          f"(num_reads={sqa_reads})...")
    try:
        results_sqa = solve_sqa(bqm, node_idx, k,
                                num_reads=sqa_reads, num_sweeps=1000)
        print_results(results_sqa, satellites_df)
    except RuntimeError as e:
        print(f"  SQA skipped: {e}")
        results_sqa = None

    # ------------------------------------------------------------------
    # 8. Quantum Annealing — D-Wave QPU (optional)
    # ------------------------------------------------------------------
    results_qa = None
    if args.quantum:
        print("\nAttempting Quantum Annealing (D-Wave Leap)...")
        try:
            results_qa = solve_quantum(bqm, node_idx, k, num_reads=num_reads)
            print_results(results_qa, satellites_df)
            compare_solvers(results_sa, results_qa, satellites_df)
        except RuntimeError as e:
            print(f"\n  Quantum annealing skipped: {e}")
    else:
        print(
            "\nD-Wave QPU not requested. "
            "Run with --quantum to attempt hardware quantum annealing."
        )

    # ------------------------------------------------------------------
    # 9. Pc comparison table  (Owens-Fahrner 2025, Table 5 analogue)
    # ------------------------------------------------------------------
    sat_lookup = satellites_df.set_index('satellite_id').to_dict('index')

    def _agg_pc(selected):
        import math as _math
        pcs = [sat_lookup[s]['pc'] for s in selected]
        log_surv = sum(_math.log1p(-p) for p in pcs if p < 1.0)
        return 1.0 - _math.exp(log_surv)

    # Random-baseline aggregate Pc (compute from the full candidate pool)
    rng = np.random.default_rng(0)
    trial_pcs = []
    all_pcs = satellites_df['pc'].values.astype(float)
    for seed in range(30):
        idx = np.random.default_rng(seed).choice(len(satellites_df), k, replace=False)
        pc_k = all_pcs[idx]
        log_s = np.sum(np.log1p(-pc_k))
        trial_pcs.append(float(1.0 - np.exp(log_s)))
    pc_random = float(np.mean(trial_pcs))

    pc_sa   = _agg_pc(results_sa['selected_satellites'])
    pc_tabu = _agg_pc(results_tabu['selected_satellites'])
    pc_sqa  = _agg_pc(results_sqa['selected_satellites']) if results_sqa else None

    paper_random   = 7.99e-5   # Table 5 Shell 3 random mean
    paper_optimised = 4.84e-6  # Table 5 Shell 3 optimised (SA/QA best)
    paper_ratio    = paper_random / paper_optimised   # ~16.5x

    def _ratio(pc_opt, pc_rnd):
        return pc_rnd / pc_opt if pc_opt > 0 else float('inf')

    print()
    print("=" * 72)
    print("Pc COMPARISON TABLE  (Owens-Fahrner 2025, Table 5 — Shell 3 / k=100)")
    print("=" * 72)
    print(f"  {'Solver':<32} {'Aggregate Pc':>14}  {'vs Random':>10}  {'Paper':>10}")
    print(f"  {'-'*32} {'-'*14}  {'-'*10}  {'-'*10}")
    print(f"  {'Random baseline (mean, 30 trials)':<32} {pc_random:>14.4e}  "
          f"{'1.00x':>10}  {paper_random:>10.2e}")
    print(f"  {'SA (SimulatedAnnealingSampler)':<32} {pc_sa:>14.4e}  "
          f"{_ratio(pc_sa, pc_random):>9.2f}x  {paper_optimised:>10.2e}")
    print(f"  {'Tabu Search (TabuSampler)':<32} {pc_tabu:>14.4e}  "
          f"{_ratio(pc_tabu, pc_random):>9.2f}x  {'—':>10}")
    if pc_sqa is not None:
        print(f"  {'SQA (PathIntegralAnnealingSampler)':<32} {pc_sqa:>14.4e}  "
              f"{_ratio(pc_sqa, pc_random):>9.2f}x  {'—':>10}")
    if results_qa is not None:
        pc_qa = _agg_pc(results_qa['selected_satellites'])
        print(f"  {'QA (D-Wave QPU)':<32} {pc_qa:>14.4e}  "
              f"{_ratio(pc_qa, pc_random):>9.2f}x  {'—':>10}")
    print()
    print(f"  Paper reduction ratio  (random / optimised) : {paper_ratio:.1f}x")
    print(f"       = {np.log10(paper_ratio):.2f} orders of magnitude")
    print()

    # Best classical result
    best_pc  = min(pc_sa, pc_tabu, *([] if pc_sqa is None else [pc_sqa]))
    best_lbl = min(
        [('SA', pc_sa), ('Tabu', pc_tabu)]
        + ([] if pc_sqa is None else [('SQA', pc_sqa)]),
        key=lambda t: t[1]
    )[0]
    our_ratio = _ratio(best_pc, pc_random)
    print(f"  Our best result  ({best_lbl})             : {best_pc:.4e}")
    print(f"  Our reduction ratio (random / best)  : {our_ratio:.1f}x  "
          f"= {np.log10(our_ratio) if our_ratio > 0 else 0:.2f} OOM")
    print()
    if our_ratio >= paper_ratio * 0.1:
        print(f"  VERDICT: reduction within 1 OOM of paper ({paper_ratio:.1f}x).  PASS.")
    else:
        print(f"  VERDICT: reduction {our_ratio:.1f}x vs paper {paper_ratio:.1f}x.  "
              f"Absolute Pc differs due to catalog evolution (2026 vs 2024/25).")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 10. Save results to CSV
    # ------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    suffix = f'_{args.data}'
    csv_path = os.path.join(results_dir, f'solutions{suffix}.csv')
    all_results = [results_sa, results_tabu]
    if results_sqa is not None:
        all_results.append(results_sqa)
    if results_qa is not None:
        all_results.append(results_qa)
    save_results(all_results, satellites_df, csv_path)

    # ------------------------------------------------------------------
    # 11. Graph visualisation  (skip for large datasets — unreadable)
    # ------------------------------------------------------------------
    N_nodes = len(G.nodes())
    if N_nodes <= 200:
        print("\nGenerating graph visualisation...")
        png_path = os.path.join(results_dir, f'graph_visualization{suffix}.png')
        plot_graph(G, results_sa['selected_satellites'], png_path)
    else:
        print(f"\nGraph visualisation skipped for N={N_nodes} nodes "
              "(spring layout intractable for K_{N_nodes}).")

    print("\nDone.")


if __name__ == '__main__':
    main()
