"""
compute_coverage_shell3r.py  --  STEP 4
-----------------------------------------
Computes geometric land coverage scores for shell3r candidates.

Coverage proxy:
    For each candidate at each of the 4,321 SGP4 timesteps, compute
    the sub-satellite latitude and check whether it falls inside the
    target latitude band.

    lat_deg = arcsin(z / sqrt(x^2 + y^2 + z^2)) * 180 / pi
    in_band = 1  if  TARGET_BAND[0] <= lat_deg <= TARGET_BAND[1]
              0  otherwise

    coverage_raw_i  = sum(in_band) / 4321
    coverage_norm_i = coverage_raw_i / max(coverage_raw_j)

Limitations (documented):
    - Geometric fraction of time in latitude band only.
    - No minimum elevation angle constraint.
    - No SNR or link-budget model.
    - For circular orbits at fixed inclination, time-averaged coverage
      converges to the same value regardless of RAAN after ~10 orbits.
      Per-candidate differentiation comes from the finite 3-day window
      and the specific phase (mean_anomaly=0 for all candidates).

GATE:
    ratio = max(coverage_norm) / (min(coverage_norm) + 1e-15)
    If ratio < 1.05:
        Retry with narrower band (30, 40) degN.
        If still < 1.05 after retry: print WARNING and continue.
    Report the band that gives the best ratio.

Inputs:
    data/shell3r_candidates.csv    (norad_id, raan_deg)
    data/propagated_shell3r.csv    (norad_id, timestep, x_km, y_km, z_km)

Output:
    data/shell3r_coverage.csv
    Columns: norad_id, raan_deg, coverage_raw, coverage_norm

Usage:
    python src/compute_coverage_shell3r.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent.parent / "data"
CAND_CSV    = DATA_DIR / "shell3r_candidates.csv"
PROP_CSV    = DATA_DIR / "propagated_shell3r.csv"
OUTPUT_CSV  = DATA_DIR / "shell3r_coverage.csv"

# ---------------------------------------------------------------------------
# Coverage parameters
# ---------------------------------------------------------------------------
TARGET_BAND_PRIMARY  = (20.0, 50.0)  # degrees N — primary band
TARGET_BAND_NARROW   = (30.0, 40.0)  # degrees N — retry if ratio < threshold
RATIO_THRESHOLD      = 1.05          # min max/min ratio to pass gate


# ---------------------------------------------------------------------------
# Coverage computation for a given band
# ---------------------------------------------------------------------------

def compute_coverage(
    df_cand: pd.DataFrame,
    df_prop: pd.DataFrame,
    band: tuple[float, float],
) -> pd.DataFrame:
    """
    Compute coverage_raw and coverage_norm for the given latitude band.

    Returns DataFrame with columns: norad_id, raan_deg, coverage_raw, coverage_norm.
    """
    lat_min_rad = math.radians(band[0])
    lat_max_rad = math.radians(band[1])

    x = df_prop['x_km'].values
    y = df_prop['y_km'].values
    z = df_prop['z_km'].values
    r = np.sqrt(x**2 + y**2 + z**2)

    valid   = r > 1.0   # filter SGP4 failure rows (0,0,0)
    lat_rad = np.where(
        valid,
        np.arcsin(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0)),
        np.nan,
    )
    in_band = (lat_rad >= lat_min_rad) & (lat_rad <= lat_max_rad)

    df_work = pd.DataFrame({
        'norad_id': df_prop['norad_id'].values,
        'valid':    valid,
        'in_band':  in_band,
    })

    grp          = df_work.groupby('norad_id')
    valid_steps  = grp['valid'].sum()
    inband_steps = grp['in_band'].sum()
    cov_raw_s    = (inband_steps / valid_steps.replace(0, np.nan)).fillna(0.0)
    cov_raw_s.name = 'coverage_raw'

    df_out = df_cand[['norad_id', 'raan_deg']].copy()
    df_out = df_out.merge(cov_raw_s.reset_index(), on='norad_id', how='left')
    df_out['coverage_raw'] = df_out['coverage_raw'].fillna(0.0)

    cmax = float(df_out['coverage_raw'].max())
    df_out['coverage_norm'] = df_out['coverage_raw'] / cmax if cmax > 0 else 0.0

    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("Shell3r Coverage Computation")
    print("=" * 62)

    for p in [CAND_CSV, PROP_CSV]:
        if not p.exists():
            print(f"\n  ERROR: {p.name} not found.")
            print("  Run STEP 1 and STEP 2 first.")
            sys.exit(1)

    df_cand = pd.read_csv(CAND_CSV)
    print(f"\n  Candidates : {len(df_cand)}")

    print(f"  Loading {PROP_CSV.name} ...")
    df_prop = pd.read_csv(PROP_CSV, dtype={
        'norad_id': np.int32, 'timestep': np.int32,
        'x_km': np.float32,   'y_km': np.float32,   'z_km': np.float32,
    })
    print(f"  Propagated rows : {len(df_prop):,}")

    T_actual = int(df_prop['timestep'].max()) + 1
    print(f"  Timesteps per sat: {T_actual}")

    # --- Primary band --------------------------------------------------------
    band     = TARGET_BAND_PRIMARY
    df_cov   = compute_coverage(df_cand, df_prop, band)
    cmax     = float(df_cov['coverage_raw'].max())
    cmin_nz  = float(df_cov.loc[df_cov['coverage_raw'] > 0, 'coverage_raw'].min()
                     if (df_cov['coverage_raw'] > 0).any() else 0.0)
    ratio    = cmax / (cmin_nz + 1e-15)

    print(f"\n  Primary band ({band[0]:.0f}N-{band[1]:.0f}N deg):")
    print(f"    coverage_raw : min={df_cov['coverage_raw'].min():.4f}  "
          f"max={df_cov['coverage_raw'].max():.4f}  "
          f"mean={df_cov['coverage_raw'].mean():.4f}")
    print(f"    Max/min ratio: {ratio:.3f}")

    chosen_band = band

    if ratio < RATIO_THRESHOLD:
        print(f"  WARNING: ratio {ratio:.3f} < {RATIO_THRESHOLD} — trying narrower band.")
        band2     = TARGET_BAND_NARROW
        df_cov2   = compute_coverage(df_cand, df_prop, band2)
        cmax2     = float(df_cov2['coverage_raw'].max())
        cmin2_nz  = float(df_cov2.loc[df_cov2['coverage_raw'] > 0, 'coverage_raw'].min()
                          if (df_cov2['coverage_raw'] > 0).any() else 0.0)
        ratio2    = cmax2 / (cmin2_nz + 1e-15)

        print(f"\n  Narrow band ({band2[0]:.0f}N-{band2[1]:.0f}N deg):")
        print(f"    coverage_raw : min={df_cov2['coverage_raw'].min():.4f}  "
              f"max={df_cov2['coverage_raw'].max():.4f}  "
              f"mean={df_cov2['coverage_raw'].mean():.4f}")
        print(f"    Max/min ratio: {ratio2:.3f}")

        if ratio2 >= RATIO_THRESHOLD:
            df_cov       = df_cov2
            chosen_band  = band2
            ratio        = ratio2
            print(f"  Using narrow band {band2[0]:.0f}N-{band2[1]:.0f}N "
                  f"(ratio {ratio2:.3f} >= {RATIO_THRESHOLD}).")
        else:
            print(f"  WARNING: both bands below threshold. "
                  f"Using primary band {band[0]:.0f}N-{band[1]:.0f}N.")
            print("  Documented limitation: time-averaged coverage at fixed inclination")
            print("  is nearly uniform across RAAN for the 3-day window used.")

    # --- Summary -------------------------------------------------------------
    print(f"\n  Coverage band  : {chosen_band[0]:.0f}N - {chosen_band[1]:.0f}N deg")
    print(f"  Mean coverage_raw  : {df_cov['coverage_raw'].mean()*100:.2f}%")
    print(f"  Range              : [{df_cov['coverage_raw'].min()*100:.2f}%, "
          f"{df_cov['coverage_raw'].max()*100:.2f}%]")
    print(f"  Max/min ratio      : {ratio:.3f}")

    n_nonzero = int((df_cov['coverage_raw'] > 0).sum())
    print(f"  Sats coverage > 0  : {n_nonzero}/{len(df_cov)}")

    # --- Save ----------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_cov[['norad_id', 'raan_deg', 'coverage_raw', 'coverage_norm']].to_csv(
        OUTPUT_CSV, index=False, float_format='%.6f'
    )
    print(f"\n  Saved {len(df_cov)} rows to: {OUTPUT_CSV.name}")

    # --- Gate ----------------------------------------------------------------
    assert len(df_cov) == len(df_cand), (
        f"GATE FAILED: expected {len(df_cand)} rows, got {len(df_cov)}"
    )

    gate_msg = "PASS" if ratio >= RATIO_THRESHOLD else "WARN (low differentiation)"
    print(f"\n  GATE: {gate_msg}  |  ratio = {ratio:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()
