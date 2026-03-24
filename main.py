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
        choices=['sample', 'real'],
        help="Dataset to use: 'sample' (synthetic, default) or 'real' (SGP4-derived Pc + TLE coverage)"
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
    num_reads = args.num_reads

    # ------------------------------------------------------------------
    # 1. Load satellite data
    # ------------------------------------------------------------------
    if args.data == 'real':
        data_file = 'real_satellites.csv'
    else:
        data_file = 'sample_satellites.csv'
    data_path = os.path.join(os.path.dirname(__file__), 'data', data_file)
    print(f"\nLoading satellite data from: {data_path}")
    satellites_df = pd.read_csv(data_path)
    satellites_df['satellite_id'] = satellites_df['satellite_id'].astype(str)
    print(satellites_df[['satellite_id', 'pc', 'coverage']].to_string(index=False))
    print(f"\n  {len(satellites_df)} candidate satellites loaded.")
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
    # 7. Quantum Annealing (optional)
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
            "\nQuantum annealing not requested. "
            "Run with --quantum to attempt QPU solving via D-Wave Leap."
        )

    # ------------------------------------------------------------------
    # 8. Save results to CSV
    # ------------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    suffix = f'_{args.data}'
    csv_path = os.path.join(results_dir, f'solutions{suffix}.csv')
    all_results = [results_sa, results_tabu]
    if results_qa is not None:
        all_results.append(results_qa)
    save_results(all_results, satellites_df, csv_path)

    # ------------------------------------------------------------------
    # 9. Graph visualisation
    # ------------------------------------------------------------------
    print("\nGenerating graph visualisation...")
    png_path = os.path.join(results_dir, f'graph_visualization{suffix}.png')
    plot_graph(G, results_sa['selected_satellites'], png_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
