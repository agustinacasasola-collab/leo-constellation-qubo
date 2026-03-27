"""
build_arnas_dataset.py
----------------------
Prepares data/arnas_candidates.csv for the QUBO pipeline from
the raw candidates_pc.csv produced by compute_pc.py --synthetic.

Coverage metric
---------------
The paper (Owens-Fahrner 2025, Section 4) computes coverage as the
fraction of 3-day SGP4 simulation timesteps where the satellite has
valid access to LEO catalog objects.  For Shell 3 (i=30°, 550 km)
that full simulation is expensive; as a placeholder we use the
inclination-band proxy:

    coverage = |sin(inclination_rad)|

For i = 30°:  sin(30°) = 0.5  (all 1,656 candidates identical)

This is a simplified metric — every candidate receives the same
coverage score, so the QUBO optimiser's selection is driven
exclusively by the collision-risk term Pc_n.  Replace with
per-satellite 3-day access fractions (Owens-Fahrner 2025 Section 4)
to fully replicate Table 5 of the paper.

Output columns (arnas_candidates.csv)
--------------------------------------
satellite_id      — NORAD ID as string
pc                — aggregate collision probability Pc_n  (Chan 2D, product formula)
coverage          — inclination proxy  |sin(30°)| = 0.5
raan_deg          — RAAN from TLE (degrees)
mean_anomaly_deg  — mean anomaly from TLE (degrees)

Usage
-----
    python src/build_arnas_dataset.py
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
RESULTS_DIR   = ROOT_DIR / "results"

INPUT_CSV     = DATA_DIR / "candidates_pc.csv"
OUTPUT_CSV    = DATA_DIR / "arnas_candidates.csv"
HISTOGRAM_PNG = RESULTS_DIR / "pc_distribution_arnas.png"

# ---------------------------------------------------------------------------
# Orbital parameters (Shell 3 — Arnas 2021)
# ---------------------------------------------------------------------------
INCLINATION_DEG = 30.0
# Coverage proxy: fraction of Earth surface latitude band within ±i degrees
# = |sin(i)|.  For i=30°: sin(30°) = 0.5 exactly.
COVERAGE_PROXY  = abs(math.sin(math.radians(INCLINATION_DEG)))


# ---------------------------------------------------------------------------
# Histogram helper
# ---------------------------------------------------------------------------

def _plot_histogram(pcs: np.ndarray, n_zero: int, output_path: Path) -> None:
    """
    Two-panel figure:
      Left  — log₁₀(Pc) histogram for non-zero candidates.
      Right — stacked bar chart of risk buckets.
    """
    nz      = pcs[pcs > 0]
    n_total = len(pcs)
    n_low   = int(((nz > 0) & (nz <= 1e-6)).sum())
    n_med   = int(((nz > 1e-6) & (nz <= 1e-4)).sum())
    n_high  = int((nz > 1e-4).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Arnas Shell 3 — Aggregate Collision Probability Distribution\n"
        "1,656 candidates  |  550 km / 30°  |  Chan 2D formula (σ = 100 m)",
        fontsize=11,
    )

    # Left: log-scale histogram of non-zero Pc values
    ax = axes[0]
    if len(nz) > 0:
        ax.hist(np.log10(nz), bins=40,
                color='steelblue', edgecolor='white', linewidth=0.4)
    ax.set_xlabel("log₁₀(Pc)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Non-zero Pc  (n = {len(nz):,})", fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Right: risk bucket bar chart
    ax2    = axes[1]
    labels = [
        "Pc = 0\n(safest)",
        "0 < Pc <= 1e-6\n(low risk)",
        "1e-6 < Pc <= 1e-4\n(medium risk)",
        "Pc > 1e-4\n(high risk)",
    ]
    counts = [n_zero, n_low, n_med, n_high]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    bars = ax2.bar(labels, counts, color=colors, edgecolor='white')
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{cnt}\n({100 * cnt / n_total:.1f}%)",
                ha='center', va='bottom', fontsize=9,
            )

    ax2.set_ylabel("Number of candidates", fontsize=11)
    ax2.set_title("Pc risk buckets", fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Arnas Shell 3 Dataset Builder")
    print(f"  Input    : {INPUT_CSV.name}")
    print(f"  Output   : {OUTPUT_CSV.name}")
    print(f"  Coverage : |sin({INCLINATION_DEG}°)| = {COVERAGE_PROXY:.4f}  "
          f"(inclination proxy — placeholder)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load candidates_pc.csv
    # ------------------------------------------------------------------
    if not INPUT_CSV.exists():
        print(f"\nERROR: {INPUT_CSV} not found.")
        print("Run 'python src/compute_pc.py --synthetic' first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"\nLoaded {len(df):,} candidates from {INPUT_CSV.name}")
    print(f"  Columns : {df.columns.tolist()}")

    # ------------------------------------------------------------------
    # 2. Rename columns and add coverage
    #    graph_builder.py requires: satellite_id, pc, coverage
    # ------------------------------------------------------------------
    df = df.rename(columns={'norad_id': 'satellite_id', 'Pc_n': 'pc'})
    df['satellite_id'] = df['satellite_id'].astype(str)
    df['coverage']     = COVERAGE_PROXY

    df_out = df[['satellite_id', 'pc', 'coverage',
                 'raan_deg', 'mean_anomaly_deg']].copy()

    # ------------------------------------------------------------------
    # 3. Summary statistics
    # ------------------------------------------------------------------
    pcs       = df_out['pc'].values
    n_total   = len(pcs)
    n_zero    = int((pcs == 0).sum())
    n_nonzero = int((pcs > 0).sum())
    n_low     = int(((pcs > 0) & (pcs <= 1e-6)).sum())
    n_med     = int(((pcs > 1e-6) & (pcs <= 1e-4)).sum())
    n_high    = int((pcs > 1e-4).sum())

    print()
    print("=" * 65)
    print("Pc DISTRIBUTION SUMMARY")
    print("=" * 65)
    print(f"  Total candidates           : {n_total:,}")
    print(f"  Pc = 0  (safest slots)     : {n_zero:,}  ({100 * n_zero / n_total:.1f}%)")
    print(f"  0 < Pc <= 1e-6             : {n_low:,}  ({100 * n_low / n_total:.1f}%)")
    print(f"  1e-6 < Pc <= 1e-4          : {n_med:,}  ({100 * n_med / n_total:.1f}%)")
    print(f"  Pc > 1e-4  (highest risk)  : {n_high:,}  ({100 * n_high / n_total:.1f}%)")

    if n_nonzero > 0:
        nz = pcs[pcs > 0]
        print()
        print(f"  Non-zero Pc statistics (n = {n_nonzero:,}):")
        print(f"    Min    : {nz.min():.4e}")
        print(f"    Max    : {pcs.max():.4e}")
        print(f"    Mean   : {pcs.mean():.4e}  (over all {n_total:,})")
        print(f"    Median : {float(np.median(pcs)):.4e}  (over all {n_total:,})")
        for p in (50, 90, 95, 99):
            print(f"    p{p:02d}    : {np.percentile(pcs, p):.4e}")

    print()
    print("  Coverage (all candidates identical):")
    print(f"    |sin({INCLINATION_DEG}°)| = {COVERAGE_PROXY:.6f}")
    print("    NOTE: This is a placeholder.  Full replication of Table 5")
    print("    requires per-satellite 3-day SGP4 access fractions.")

    # ------------------------------------------------------------------
    # 4. Histogram
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_histogram(pcs, n_zero, HISTOGRAM_PNG)
    print(f"\n  Histogram saved to : {HISTOGRAM_PNG}")

    # ------------------------------------------------------------------
    # 5. Save CSV
    # ------------------------------------------------------------------
    df_out.to_csv(OUTPUT_CSV, index=False, float_format='%.6e')
    print(f"  Dataset saved to   : {OUTPUT_CSV}")
    print(f"  Shape              : {df_out.shape}")
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
