"""
compute_coverage_shell_a.py  --  Task 1  (Option B: TCA-latitude coverage)
---------------------------------------------------------------------------
Computes a population-weighted coverage score for each Shell A satellite
using the latitude at Time of Closest Approach (lat_deg) from the Pc
computation step.

Coverage model:
    v_raw[i]  = exp( -(lat_i - MU_LAT)^2 / (2 * SIGMA_LAT^2) )
    coverage_norm[i] = v_raw[i] / max(v_raw)

Interpretation: satellites whose closest conjunction occurs near MU_LAT
(30 degN, population centroid) are scored higher.  Different RAAN values
produce conjunctions at different times and latitudes, giving genuine
per-satellite differentiation.

Warning: if max / min(nonzero) < 1.1 the signal is nearly uniform.

Input:
    data/multishell_pc.csv   (columns include inc_deg, raan_deg, lat_deg)
    Filtered to inc ~ 53 deg (+/- INC_TOL_DEG).

Output:
    data/coverage_shell_a.csv
    Columns: norad_id, raan_deg, lat_deg, coverage_raw, coverage_norm

Usage:
    python src/compute_coverage_shell_a.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent.parent / "data"
PC_CSV     = DATA_DIR / "multishell_pc.csv"
OUTPUT_CSV = DATA_DIR / "coverage_shell_a.csv"

# ---------------------------------------------------------------------------
# Coverage model parameters
# ---------------------------------------------------------------------------
MU_LAT    = 30.0   # deg -- population centroid
SIGMA_LAT = 20.0   # deg -- Gaussian spread

# Shell A filter
SHELL_A_INC = 53.0
INC_TOL     = 1.0

UNIFORMITY_WARN = 1.1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Shell A Coverage  (TCA-latitude Gaussian model)")
    print(f"  Gaussian: exp(-(lat - {MU_LAT})^2 / (2 * {SIGMA_LAT}^2))")
    print("=" * 60)

    if not PC_CSV.exists():
        print(f"\n  ERROR: {PC_CSV.name} not found.")
        print("  Run 'python src/compute_pc_multishell.py' first.")
        sys.exit(1)

    # --- Load and filter to Shell A ------------------------------------------
    df_all = pd.read_csv(PC_CSV)
    mask   = np.abs(df_all['inc_deg'].values - SHELL_A_INC) <= INC_TOL
    df     = df_all[mask].reset_index(drop=True)
    print(f"\n  Loaded {len(df_all)} rows, kept {len(df)} "
          f"(inc {SHELL_A_INC}+/-{INC_TOL} deg)")

    if len(df) == 0:
        print("  ERROR: no Shell A candidates found.")
        sys.exit(1)

    if 'lat_deg' not in df.columns:
        print("  ERROR: 'lat_deg' column missing from multishell_pc.csv.")
        print("  Re-run compute_pc_multishell.py (lat_deg output was added in the")
        print("  single-shell modification).")
        sys.exit(1)

    lat_values = df['lat_deg'].values.astype(np.float64)

    # --- Gaussian coverage score ---------------------------------------------
    v_raw  = np.exp(-0.5 * ((lat_values - MU_LAT) / SIGMA_LAT) ** 2)
    v_max  = float(v_raw.max())
    v_norm = v_raw / v_max if v_max > 0 else v_raw

    # --- Uniformity check ----------------------------------------------------
    v_min_nonzero = float(v_raw[v_raw > 0].min()) if (v_raw > 0).any() else 0.0
    if v_min_nonzero > 0:
        ratio = v_max / v_min_nonzero
        if ratio < UNIFORMITY_WARN:
            print(f"\n  WARNING: coverage nearly uniform "
                  f"(max/min = {ratio:.3f} < {UNIFORMITY_WARN}).")

    # --- Print summary -------------------------------------------------------
    print(f"\n  lat_deg range    : {lat_values.min():.1f} -- {lat_values.max():.1f} deg")
    print(f"  coverage_raw     : min={v_raw.min():.4f}  max={v_raw.max():.4f}  "
          f"mean={v_raw.mean():.4f}")
    print(f"  coverage_norm    : min={v_norm.min():.4f}  max={v_norm.max():.4f}  "
          f"mean={v_norm.mean():.4f}")
    n_high = int((v_norm >= 0.5).sum())
    print(f"  Sats with norm >= 0.5: {n_high} / {len(df)}")

    # --- Save ----------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        'norad_id':     df['norad_id'],
        'raan_deg':     df['raan_deg'],
        'lat_deg':      lat_values,
        'coverage_raw': v_raw,
        'coverage_norm': v_norm,
    })
    df_out.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
    print(f"\n  Saved {len(df_out)} rows to: {OUTPUT_CSV.name}")

    # --- Gate ----------------------------------------------------------------
    assert len(df_out) == len(df), (
        f"GATE FAILED: expected {len(df)} rows, got {len(df_out)}"
    )
    ratio_str = f"{v_max / v_min_nonzero:.2f}" if v_min_nonzero > 0 else "inf"
    print(f"  max/min ratio    : {ratio_str}")
    print(f"\n  GATE PASS: {len(df_out)} coverage rows written  OK")
    print("Done.")


if __name__ == "__main__":
    main()
