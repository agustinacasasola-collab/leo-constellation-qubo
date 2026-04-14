"""
analyze_multishell.py
---------------------
Generates results/multishell_analysis.png with 3 subplots.

Subplot 1 — Pc by shell (box plot, log scale):
    X: Shell A (53°), Shell B (70°), Shell C (97.6°)
    Y: individual Pc_n (log scale)
    Validates that the three inclinations see different debris environments.

Subplot 2 — Shell distribution in best solutions (grouped bar chart):
    Groups: Random / SA / SQA
    3 bars per group (A, B, C)
    Y: number of satellites selected from that shell
    Shows whether the optimizer concentrates in one shell or distributes.

Subplot 3 — Solution quality over 50 runs (line chart, log scale):
    X: run index 0–49
    Blue : SA aggregate Pc per run
    Orange: SQA aggregate Pc per run
    Gray band: random mean ± std

Gate:
    File results/multishell_analysis.png must be saved successfully.

Inputs:
    data/multishell_pc.csv
    results/multishell_comparison.csv
    data/multishell_runs.csv              (per-run Pc values from optimize step)

Output:
    results/multishell_analysis.png

Usage:
    python src/analyze_multishell.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR      = Path(__file__).parent.parent / "data"
RESULTS_DIR   = Path(__file__).parent.parent / "results"
PC_CSV        = DATA_DIR    / "multishell_pc.csv"
COMPARISON_CSV = RESULTS_DIR / "multishell_comparison.csv"
RUNS_CSV      = DATA_DIR    / "multishell_runs.csv"
OUTPUT_PNG    = RESULTS_DIR / "multishell_analysis.png"

# Shell styling
SHELL_COLORS  = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}
SHELL_LABELS  = {"A": "Shell A (53°)", "B": "Shell B (70°)", "C": "Shell C (97.6°)"}
SOLVER_COLORS = {"Random": "#9E9E9E", "SA": "#2196F3", "SQA": "#FF5722"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_log_ylim(data: list[float], pad: float = 0.5) -> tuple[float, float]:
    """
    Safe log-axis limits for data that may contain zeros.
    Zeros are replaced with a floor of 1e-12 for display purposes.
    """
    pos = [v for v in data if v > 0]
    if not pos:
        return (1e-12, 1e-8)
    lo = min(pos) * 10 ** (-pad)
    hi = max(pos) * 10 ** pad
    return (max(lo, 1e-15), hi)


# ---------------------------------------------------------------------------
# Subplot 1: Pc by shell (box plot, log scale)
# ---------------------------------------------------------------------------

def plot_pc_by_shell(ax: plt.Axes, df_pc: pd.DataFrame) -> None:
    """Box plot of individual Pc_n values per shell."""
    shells     = ["A", "B", "C"]
    shell_data = []
    xtick_labels = []

    for lbl in shells:
        sub  = df_pc[df_pc['shell_label'] == lbl]['Pc_n'].values
        pos  = sub[sub > 0]
        # Replace zeros with a very small floor for log display
        display = np.where(sub > 0, sub, 1e-15)
        shell_data.append(display)
        inc  = df_pc[df_pc['shell_label'] == lbl]['inc_deg'].iloc[0] if len(sub) > 0 else 0
        xtick_labels.append(SHELL_LABELS[lbl])

    bp = ax.boxplot(
        shell_data,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=4, linestyle='none'),
    )
    for patch, lbl in zip(bp['boxes'], shells):
        patch.set_facecolor(SHELL_COLORS[lbl])
        patch.set_alpha(0.7)

    ax.set_yscale('log')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel('Individual Pc$_n$', fontsize=9)
    ax.set_title('Subplot 1: Pc Distribution by Shell', fontsize=10, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)

    # Annotate mean Pc per shell
    for pos_x, (lbl, data) in enumerate(zip(shells, shell_data), start=1):
        pos_data = data[data > 1e-14]
        if len(pos_data) > 0:
            mean_val = float(np.mean(pos_data))
            ax.text(pos_x, mean_val, f'μ={mean_val:.1e}',
                    ha='center', va='bottom', fontsize=7, color='black')


# ---------------------------------------------------------------------------
# Subplot 2: Shell distribution in best solutions
# ---------------------------------------------------------------------------

def plot_shell_distribution(ax: plt.Axes, df_cmp: pd.DataFrame) -> None:
    """Grouped bar chart of shell A/B/C counts in best solution per solver."""
    solvers = ["Random", "SA", "SQA"]
    shells  = ["A", "B", "C"]

    # Load data — df_cmp has columns solver, shell_A_selected, shell_B_selected, ...
    cmp_dict = {}
    for _, row in df_cmp.iterrows():
        cmp_dict[row['solver']] = {
            'A': int(row.get('shell_A_selected', 0)),
            'B': int(row.get('shell_B_selected', 0)),
            'C': int(row.get('shell_C_selected', 0)),
        }

    n_solvers  = len(solvers)
    n_shells   = len(shells)
    bar_width  = 0.25
    group_gap  = 1.0
    x_centers  = np.arange(n_solvers) * group_gap

    for s_i, shell_lbl in enumerate(shells):
        offsets = (s_i - (n_shells - 1) / 2.0) * bar_width
        counts  = [cmp_dict.get(solver, {}).get(shell_lbl, 0) for solver in solvers]
        ax.bar(
            x_centers + offsets, counts,
            width=bar_width,
            color=SHELL_COLORS[shell_lbl],
            alpha=0.8,
            label=SHELL_LABELS[shell_lbl],
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(solvers, fontsize=9)
    ax.set_ylabel('Satellites selected', fontsize=9)
    ax.set_title('Subplot 2: Shell Distribution in Best Solution', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Subplot 3: Solution quality over 50 runs
# ---------------------------------------------------------------------------

def plot_solution_quality(ax: plt.Axes, df_runs: pd.DataFrame) -> None:
    """Line chart of aggregate Pc per run for SA and SQA, with random band."""
    runs    = df_runs['run'].values
    sa_pcs  = df_runs['sa_pc'].values
    sqa_pcs = df_runs['sqa_pc'].values
    rnd_pcs = df_runs['random_pc'].values

    rnd_mean = float(np.mean(rnd_pcs))
    rnd_std  = float(np.std(rnd_pcs))

    # Replace zeros with a small floor for log-scale display
    floor    = 1e-15
    sa_disp  = np.where(sa_pcs  > 0, sa_pcs,  floor)
    sqa_disp = np.where(sqa_pcs > 0, sqa_pcs, floor)
    rnd_lo   = max(rnd_mean - rnd_std, floor)
    rnd_hi   = max(rnd_mean + rnd_std, floor)

    # Gray random band
    ax.axhspan(rnd_lo, rnd_hi, alpha=0.25, color='gray', label=f'Random mean±std')
    ax.axhline(max(rnd_mean, floor), color='gray', linestyle='--', linewidth=1)

    ax.plot(runs, sa_disp,  color='#2196F3', linewidth=1.2, label='SA')
    ax.plot(runs, sqa_disp, color='#FF5722', linewidth=1.2, label='SQA')

    ax.set_yscale('log')
    all_vals = list(sa_disp) + list(sqa_disp) + [rnd_lo, rnd_hi]
    lo, hi   = _safe_log_ylim(all_vals)
    ax.set_ylim(lo, hi)

    ax.set_xlabel('Run index', fontsize=9)
    ax.set_ylabel('Aggregate Pc', fontsize=9)
    ax.set_title('Subplot 3: Solution Quality over 50 Runs', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Multi-Shell Analysis - generating multishell_analysis.png")
    print("=" * 65)

    # --- Prerequisite checks -----------------------------------------------
    missing = []
    for p in [PC_CSV, COMPARISON_CSV, RUNS_CSV]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        print(f"\n  ERROR: missing input files: {missing}")
        print("  Run the pipeline steps 1-4 first.")
        sys.exit(1)

    df_pc  = pd.read_csv(PC_CSV)
    df_cmp = pd.read_csv(COMPARISON_CSV)
    df_runs = pd.read_csv(RUNS_CSV)

    print(f"  Loaded {len(df_pc)} candidates from {PC_CSV.name}")
    print(f"  Loaded {len(df_cmp)} solver rows from {COMPARISON_CSV.name}")
    print(f"  Loaded {len(df_runs)} run rows from {RUNS_CSV.name}")

    # --- Build figure -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Multi-Shell Walker Constellation — Pc-Only QUBO Optimisation',
        fontsize=12, fontweight='bold', y=1.01
    )

    plot_pc_by_shell(axes[0], df_pc)
    plot_shell_distribution(axes[1], df_cmp)
    plot_solution_quality(axes[2], df_runs)

    plt.tight_layout()

    # --- Save ---------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    size_kb = OUTPUT_PNG.stat().st_size / 1024
    print(f"\n  GATE PASS: saved {OUTPUT_PNG.name}  ({size_kb:.0f} KB)  OK")
    print("Done.")


if __name__ == "__main__":
    main()
