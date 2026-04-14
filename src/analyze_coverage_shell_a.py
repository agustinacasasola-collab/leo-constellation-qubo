"""
analyze_coverage_shell_a.py  --  Task 3
-----------------------------------------
Generates results/coverage_shell_a_analysis.png  (3 subplots).

Subplot 1 -- Pc vs coverage scatter
    All 200 Shell A candidates as grey dots.
    SA best-solution satellites highlighted in blue.
    SQA best-solution satellites highlighted in orange.
    X: coverage_norm, Y: Pc_n (log scale)

Subplot 2 -- RAAN vs coverage_norm
    All 200 candidates coloured by solver selection (grey / SA / SQA).
    X: raan_deg (0-360), Y: coverage_norm (0-1)
    Shows whether the optimiser clusters around high-coverage RAAN values.

Subplot 3 -- Aggregate Pc over 50 runs (convergence)
    Blue  : SA aggregate Pc per run
    Orange: SQA aggregate Pc per run
    Grey band: random mean +/- std
    Y axis: log scale

Gate:
    results/coverage_shell_a_analysis.png must be written successfully.

Inputs:
    data/coverage_shell_a.csv
    data/multishell_pc.csv  or  data/shell_a_pc.csv   (Pc values)
    results/coverage_shell_a_comparison.csv
    results/coverage_shell_a_SA_best.csv
    results/coverage_shell_a_SQA_best.csv
    data/coverage_shell_a_runs.csv

Output:
    results/coverage_shell_a_analysis.png

Usage:
    python src/analyze_coverage_shell_a.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR     = Path(__file__).parent.parent / "data"
RESULTS_DIR  = Path(__file__).parent.parent / "results"

COVERAGE_CSV   = DATA_DIR    / "coverage_shell_a.csv"
SHELL_A_PC_CSV = DATA_DIR    / "shell_a_pc.csv"
MULTISHELL_CSV = DATA_DIR    / "multishell_pc.csv"
COMPARISON_CSV = RESULTS_DIR / "coverage_shell_a_comparison.csv"
SA_BEST_CSV    = RESULTS_DIR / "coverage_shell_a_SA_best.csv"
SQA_BEST_CSV   = RESULTS_DIR / "coverage_shell_a_SQA_best.csv"
RUNS_CSV       = DATA_DIR    / "coverage_shell_a_runs.csv"
OUTPUT_PNG     = RESULTS_DIR / "coverage_shell_a_analysis.png"

SHELL_A_INC = 53.0
INC_TOL     = 1.0

COLOR_ALL = "#BDBDBD"
COLOR_SA  = "#2196F3"
COLOR_SQA = "#FF5722"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_log_ylim(data: list, pad: float = 0.5) -> tuple:
    pos = [v for v in data if v > 0]
    if not pos:
        return (1e-12, 1e-8)
    lo = min(pos) * 10 ** (-pad)
    hi = max(pos) * 10 ** pad
    return (max(lo, 1e-15), hi)


def load_pc(df_cov: pd.DataFrame) -> pd.DataFrame:
    """
    Load Pc values and merge with coverage dataframe on raan_deg.
    Returns merged dataframe with columns: norad_id, raan_deg,
    coverage_raw, coverage_norm, Pc_n.
    """
    if SHELL_A_PC_CSV.exists():
        df_pc = pd.read_csv(SHELL_A_PC_CSV)
        pc_col = 'Pc_n' if 'Pc_n' in df_pc.columns else df_pc.columns[-1]
    elif MULTISHELL_CSV.exists():
        df_all = pd.read_csv(MULTISHELL_CSV)
        mask   = np.abs(df_all['inc_deg'].values - SHELL_A_INC) <= INC_TOL
        df_pc  = df_all[mask].reset_index(drop=True)
        pc_col = 'Pc_n'
    else:
        return df_cov.assign(Pc_n=0.0)

    df_pc  = df_pc.sort_values('raan_deg').reset_index(drop=True)
    df_cov = df_cov.sort_values('raan_deg').reset_index(drop=True)

    if len(df_pc) == len(df_cov):
        df_merged = df_cov.copy()
        df_merged['Pc_n'] = df_pc[pc_col].values
    else:
        df_pc['raan_key']  = (df_pc['raan_deg'] * 100).round().astype(int)
        df_cov2            = df_cov.copy()
        df_cov2['raan_key'] = (df_cov2['raan_deg'] * 100).round().astype(int)
        df_merged = df_cov2.merge(
            df_pc[['raan_key', pc_col]].rename(columns={pc_col: 'Pc_n'}),
            on='raan_key', how='left'
        ).drop(columns='raan_key')
        df_merged['Pc_n'] = df_merged['Pc_n'].fillna(0.0)

    return df_merged


# ---------------------------------------------------------------------------
# Subplot 1: Pc vs coverage scatter
# ---------------------------------------------------------------------------

def plot_pc_vs_coverage(ax, df_all: pd.DataFrame,
                        sa_norads: set, sqa_norads: set) -> None:
    mask_sa  = df_all['norad_id'].isin(sa_norads)
    mask_sqa = df_all['norad_id'].isin(sqa_norads)
    mask_bg  = ~(mask_sa | mask_sqa)

    pc_floor = 1e-15

    def yval(v):
        return max(float(v), pc_floor)

    y_all = [yval(v) for v in df_all['Pc_n']]
    x_all = df_all['coverage_norm'].tolist()

    ax.scatter(
        [x_all[i] for i in range(len(x_all)) if mask_bg.iloc[i]],
        [y_all[i] for i in range(len(y_all)) if mask_bg.iloc[i]],
        c=COLOR_ALL, s=18, alpha=0.6, label='All candidates', zorder=1,
    )
    ax.scatter(
        [x_all[i] for i in range(len(x_all)) if mask_sa.iloc[i]],
        [y_all[i] for i in range(len(y_all)) if mask_sa.iloc[i]],
        c=COLOR_SA, s=30, alpha=0.85, label='SA best', zorder=3,
    )
    ax.scatter(
        [x_all[i] for i in range(len(x_all)) if mask_sqa.iloc[i]],
        [y_all[i] for i in range(len(y_all)) if mask_sqa.iloc[i]],
        c=COLOR_SQA, s=30, alpha=0.85, marker='^', label='SQA best', zorder=3,
    )

    ax.set_yscale('log')
    lo, hi = safe_log_ylim(y_all)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Coverage norm (20-50 degN fraction)', fontsize=9)
    ax.set_ylabel('Individual Pc$_n$', fontsize=9)
    ax.set_title('Subplot 1: Pc vs Coverage', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)


# ---------------------------------------------------------------------------
# Subplot 2: RAAN vs coverage_norm
# ---------------------------------------------------------------------------

def plot_raan_vs_coverage(ax, df_all: pd.DataFrame,
                          sa_norads: set, sqa_norads: set) -> None:
    mask_sa  = df_all['norad_id'].isin(sa_norads)
    mask_sqa = df_all['norad_id'].isin(sqa_norads)
    mask_bg  = ~(mask_sa | mask_sqa)

    ax.scatter(
        df_all.loc[mask_bg, 'raan_deg'],
        df_all.loc[mask_bg, 'coverage_norm'],
        c=COLOR_ALL, s=18, alpha=0.6, label='All candidates', zorder=1,
    )
    ax.scatter(
        df_all.loc[mask_sa, 'raan_deg'],
        df_all.loc[mask_sa, 'coverage_norm'],
        c=COLOR_SA, s=30, alpha=0.85, label='SA best', zorder=3,
    )
    ax.scatter(
        df_all.loc[mask_sqa, 'raan_deg'],
        df_all.loc[mask_sqa, 'coverage_norm'],
        c=COLOR_SQA, s=30, alpha=0.85, marker='^', label='SQA best', zorder=3,
    )

    ax.set_xlabel('RAAN (deg)', fontsize=9)
    ax.set_ylabel('Coverage norm', fontsize=9)
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Subplot 2: RAAN vs Coverage', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Subplot 3: Aggregate Pc over 50 runs
# ---------------------------------------------------------------------------

def plot_convergence(ax, df_runs: pd.DataFrame) -> None:
    runs    = df_runs['run'].values
    sa_pcs  = df_runs['sa_pc'].values
    sqa_pcs = df_runs['sqa_pc'].values
    rnd_pcs = df_runs['random_pc'].values

    rnd_mean = float(np.mean(rnd_pcs))
    rnd_std  = float(np.std(rnd_pcs))
    floor    = 1e-15

    sa_disp  = np.where(sa_pcs  > 0, sa_pcs,  floor)
    sqa_disp = np.where(sqa_pcs > 0, sqa_pcs, floor)
    rnd_lo   = max(rnd_mean - rnd_std, floor)
    rnd_hi   = max(rnd_mean + rnd_std, floor)

    ax.axhspan(rnd_lo, rnd_hi, alpha=0.2, color='gray', label='Random mean+/-std')
    ax.axhline(max(rnd_mean, floor), color='gray', linestyle='--', linewidth=1.0)
    ax.plot(runs, sa_disp,  color=COLOR_SA,  linewidth=1.3, label='SA')
    ax.plot(runs, sqa_disp, color=COLOR_SQA, linewidth=1.3, label='SQA')

    ax.set_yscale('log')
    all_vals = list(sa_disp) + list(sqa_disp) + [rnd_lo, rnd_hi]
    lo, hi   = safe_log_ylim(all_vals)
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
    print("Shell A Coverage Analysis  -- generating analysis PNG")
    print("=" * 65)

    # Prerequisites
    missing = []
    for p in [COVERAGE_CSV, COMPARISON_CSV, SA_BEST_CSV, SQA_BEST_CSV, RUNS_CSV]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        print(f"\n  ERROR: missing input files: {missing}")
        print("  Run Tasks 1 and 2 first.")
        sys.exit(1)

    df_cov = pd.read_csv(COVERAGE_CSV)
    print(f"  Coverage     : {len(df_cov)} rows")

    df_all = load_pc(df_cov)
    print(f"  Merged Pc+cov: {len(df_all)} rows")

    df_sa_best  = pd.read_csv(SA_BEST_CSV)
    df_sqa_best = pd.read_csv(SQA_BEST_CSV)
    df_runs     = pd.read_csv(RUNS_CSV)

    print(f"  SA best      : {len(df_sa_best)} selected satellites")
    print(f"  SQA best     : {len(df_sqa_best)} selected satellites")
    print(f"  Runs         : {len(df_runs)} rows")

    sa_norads  = set(df_sa_best['norad_id'].tolist())
    sqa_norads = set(df_sqa_best['norad_id'].tolist())

    # --- Build figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Shell A Bi-Objective QUBO  (Pc + Coverage 20-50 degN)',
        fontsize=12, fontweight='bold', y=1.01,
    )

    plot_pc_vs_coverage(axes[0], df_all, sa_norads, sqa_norads)
    plot_raan_vs_coverage(axes[1], df_all, sa_norads, sqa_norads)
    plot_convergence(axes[2], df_runs)

    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    size_kb = OUTPUT_PNG.stat().st_size / 1024
    print(f"\n  GATE PASS: saved {OUTPUT_PNG.name}  ({size_kb:.0f} KB)  OK")
    print("Done.")


if __name__ == "__main__":
    main()
