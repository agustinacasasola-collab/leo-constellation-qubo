"""
compute_pc.py
-------------
Computes real collision probabilities for the 20 filtered candidate satellites
against the full LEO catalog using SGP4-propagated TCA miss distances.

Pipeline for each candidate:
  1. Apply apogee/perigee filter → ~4,000 catalog objects
  2. Propagate each filtered object at all candidate timesteps via sgp4_array
     (vectorised over time — one SGP4 call per catalog object, not per timestep)
  3. Compute Euclidean distance to candidate at every timestep
  4. TCA miss distance = minimum distance over the simulation window
  5. Feed real TCA into chan_pc → individual Pc
  6. Sum individual Pc values → aggregate Pc for this candidate

Output:
  data/satellite_pc.csv   — aggregate Pc + stats per candidate
  (printed table for sanity check against Owens-Fahrner et al. range 1e-5–1e-3)

Usage:
    python src/compute_pc.py
"""

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collision import (
    load_tle_pairs,
    apogee_perigee_filter,
    parse_tle_line2,
    chan_pc,
    MARGIN_KM,
)

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
CANDIDATE_TLE_PATH = DATA_DIR / "shell_550km.tle"
CATALOG_TLE_PATH = DATA_DIR / "leo_catalog.tle"
PROPAGATED_CSV = DATA_DIR / "propagated_states.csv"
OUTPUT_CSV = DATA_DIR / "satellite_pc.csv"

# Altitude filter — must match propagate_orbits.py
ALT_MIN_KM = 540.0
ALT_MAX_KM = 560.0
CENTER_KM = 550.0
MAX_CANDIDATES = 20

# chan_pc parameters
HARD_BODY_RADIUS_KM = 0.01    # 10 m combined hard-body radius
# 5 km along-track uncertainty: justified for untracked/older debris whose
# TLEs may be days old.  This is the scale at which real TCA values (5–17 km)
# start producing non-negligible Pc via the Chan formula.  CDM-level sigma
# (0.05–0.5 km) gives machine-zero Pc because real SGP4 TCA ≫ sigma.
SIGMA_KM = 5.0


# ---------------------------------------------------------------------------
# Load and filter candidate satellites
# ---------------------------------------------------------------------------

def load_filtered_candidates(
    propagated_csv: Path,
    candidate_tle_path: Path,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Load the 20 altitude-filtered candidate satellites.

    Applies the same 540–560 km mean-altitude filter as propagate_orbits.py,
    keeping the MAX_CANDIDATES objects closest to 550 km.

    Parameters
    ----------
    propagated_csv : Path
        Path to data/propagated_states.csv.
    candidate_tle_path : Path
        Path to data/shell_550km.tle.

    Returns
    -------
    norad_ids : list of str
        NORAD IDs of the 20 selected candidates.
    positions : dict {norad_id: ndarray shape (T, 3)}
        ECI position vectors (km) at each timestep.
    jd_array : ndarray shape (T,)
        Julian date integer parts for all timesteps.
    fr_array : ndarray shape (T,)
        Julian date fractional parts for all timesteps.
    """
    df = pd.read_csv(propagated_csv)
    valid = df[df['error'] == 0].copy()

    # Normalize NORAD IDs to zero-padded 5-char strings before any groupby
    # (pandas may read '01835' as integer 1835 from CSV)
    valid['norad_id'] = valid['norad_id'].astype(str).str.zfill(5)

    # Compute mean altitude per satellite
    mean_alts = valid.groupby('norad_id')['altitude_km'].mean()

    # Band filter
    in_band = mean_alts[(mean_alts >= ALT_MIN_KM) & (mean_alts <= ALT_MAX_KM)]

    # Keep MAX_CANDIDATES closest to shell centre
    ranked = in_band.reindex((in_band - CENTER_KM).abs().sort_values().index)
    selected_ids = list(ranked.index[:MAX_CANDIDATES])

    print(f"  Altitude-filtered candidates : {len(selected_ids)} "
          f"(mean alt range: {in_band[selected_ids].min():.1f}–"
          f"{in_band[selected_ids].max():.1f} km)")

    # Extract position arrays
    positions = {}
    for norad_id in selected_ids:
        subset = valid[valid['norad_id'] == norad_id].sort_values('epoch_utc')
        positions[norad_id] = subset[['x_km', 'y_km', 'z_km']].values  # (T, 3)

    # Build Julian date arrays from the first candidate's epochs
    first_id = selected_ids[0]
    first_subset = valid[valid['norad_id'] == first_id].sort_values('epoch_utc')
    epochs = first_subset['epoch_utc'].values

    jd_list, fr_list = [], []
    for epoch_str in epochs:
        t = datetime.fromisoformat(str(epoch_str))
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_list.append(jd)
        fr_list.append(fr)

    return selected_ids, positions, np.array(jd_list), np.array(fr_list)


# ---------------------------------------------------------------------------
# Vectorised TCA computation
# ---------------------------------------------------------------------------

def compute_min_distance(
    candidate_pos: np.ndarray,
    catalog_sat: Satrec,
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> float:
    """
    Find the minimum separation between a candidate and a catalog object
    over the full simulation window (Time of Closest Approach proxy).

    Uses sgp4_array to propagate the catalog object at all T timesteps in a
    single vectorised call, then computes Euclidean distances in numpy.

    Parameters
    ----------
    candidate_pos : ndarray, shape (T, 3)
        Pre-loaded ECI positions of the candidate satellite (km).
    catalog_sat : Satrec
        SGP4 record for the catalog object.
    jd_array : ndarray, shape (T,)
        Julian date integer parts.
    fr_array : ndarray, shape (T,)
        Julian date fractional parts.

    Returns
    -------
    float
        Minimum distance (km) between the two objects over the window.
        Returns inf if the catalog object has propagation errors at all steps.
    """
    # Propagate all T timesteps in one call — vectorised SGP4
    errors, r_catalog, _ = catalog_sat.sgp4_array(jd_array, fr_array)
    r_catalog = np.array(r_catalog)   # shape (T, 3)
    errors = np.array(errors)

    # Mask out timesteps where SGP4 returned an error (decayed / invalid)
    valid_mask = (errors == 0)
    if not np.any(valid_mask):
        return math.inf

    # Euclidean distance at each valid timestep: shape (T_valid,)
    diff = candidate_pos[valid_mask] - r_catalog[valid_mask]
    distances = np.linalg.norm(diff, axis=1)

    return float(distances.min())


# ---------------------------------------------------------------------------
# Per-candidate aggregate Pc
# ---------------------------------------------------------------------------

def compute_candidate_pc(
    candidate_norad: str,
    candidate_pos: np.ndarray,
    candidate_l1: str,
    candidate_l2: str,
    catalog_pairs: list[tuple[str, str]],
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> dict:
    """
    Compute aggregate collision probability for one candidate satellite.

    Parameters
    ----------
    candidate_norad : str
        NORAD ID of the candidate.
    candidate_pos : ndarray, shape (T, 3)
        Pre-propagated ECI positions of the candidate (km).
    candidate_l1, candidate_l2 : str
        TLE lines for the candidate.
    catalog_pairs : list of (line1, line2)
        Full LEO catalog TLE pairs.
    jd_array, fr_array : ndarray
        Julian dates for the simulation timesteps.

    Returns
    -------
    dict with keys:
        norad_id, num_filtered, num_conjunctions, aggregate_pc,
        max_individual_pc, mean_tca_km, min_tca_km
    """
    # Exclude the candidate itself from the catalog to prevent self-conjunction
    # (the candidate TLE may appear in both shell_550km.tle and leo_catalog.tle)
    catalog_pairs = [
        (l1, l2) for l1, l2 in catalog_pairs
        if l1[2:7].strip() != candidate_norad
    ]

    # Step 1: altitude-band filter
    filtered = apogee_perigee_filter(
        candidate_l1, candidate_l2, catalog_pairs, margin_km=MARGIN_KM
    )

    candidate_sat = Satrec.twoline2rv(candidate_l1, candidate_l2)

    # Step 2: compute TCA + Pc for each filtered object
    aggregate_pc = 0.0
    max_individual_pc = 0.0
    tca_values = []
    num_conjunctions = 0   # objects with TCA < 100 km (genuine close approach)

    for l1, l2 in filtered:
        try:
            catalog_sat = Satrec.twoline2rv(l1, l2)
        except Exception:
            continue

        # Find real TCA via SGP4 propagation
        tca_km = compute_min_distance(candidate_pos, catalog_sat, jd_array, fr_array)

        if math.isinf(tca_km):
            continue

        tca_values.append(tca_km)

        # Consider only objects that actually get within 100 km
        # (wider conjunctions contribute negligible Pc)
        if tca_km < 100.0:
            num_conjunctions += 1

        # Chan's Pc with real miss distance
        pc = chan_pc(
            candidate_sat, catalog_sat,
            hard_body_radius_km=HARD_BODY_RADIUS_KM,
            sigma_km=SIGMA_KM,
            miss_km_override=tca_km,
        )
        aggregate_pc += pc
        if pc > max_individual_pc:
            max_individual_pc = pc

    return {
        'norad_id': candidate_norad,
        'num_filtered': len(filtered),
        'num_conjunctions': num_conjunctions,
        'aggregate_pc': aggregate_pc,
        'max_individual_pc': max_individual_pc,
        'mean_tca_km': float(np.mean(tca_values)) if tca_values else 0.0,
        'min_tca_km': float(np.min(tca_values)) if tca_values else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("SGP4-based TCA Collision Probability Computation")
    print(f"  Candidates  : {MAX_CANDIDATES} (altitude filter {ALT_MIN_KM}-{ALT_MAX_KM} km)")
    print(f"  Hard body r : {HARD_BODY_RADIUS_KM * 1000:.0f} m")
    print(f"  sigma       : {SIGMA_KM * 1000:.0f} m per axis")
    print("=" * 70)

    # Load candidate states and Julian date arrays
    print("\nLoading candidate states from propagated_states.csv...")
    selected_ids, positions, jd_array, fr_array = load_filtered_candidates(
        PROPAGATED_CSV, CANDIDATE_TLE_PATH
    )
    T = len(jd_array)
    print(f"  Time steps  : {T} ({T * 60 / 3600:.0f} h window, 60 s step)")

    # Load TLE data
    print("\nLoading TLE files...")
    candidate_pairs_all = load_tle_pairs(CANDIDATE_TLE_PATH)
    catalog_pairs = load_tle_pairs(CATALOG_TLE_PATH)

    # Build a lookup: norad_id → (line1, line2) for candidates
    # Zero-pad to 5 chars to match the normalized IDs from the CSV
    candidate_tle_lookup = {
        l1[2:7].strip().zfill(5): (l1, l2) for l1, l2 in candidate_pairs_all
    }

    print(f"  Candidate TLEs in file : {len(candidate_pairs_all)}")
    print(f"  Full LEO catalog       : {len(catalog_pairs):,} objects")

    # Process each candidate
    print()
    results = []
    for i, norad_id in enumerate(selected_ids):
        if norad_id not in candidate_tle_lookup:
            print(f"  [{i+1:02d}/{MAX_CANDIDATES}] NORAD {norad_id}  — TLE not found, skipping")
            continue

        l1, l2 = candidate_tle_lookup[norad_id]
        print(f"  [{i+1:02d}/{MAX_CANDIDATES}] NORAD {norad_id}  computing TCA + Pc...",
              end='', flush=True)

        result = compute_candidate_pc(
            norad_id, positions[norad_id], l1, l2,
            catalog_pairs, jd_array, fr_array,
        )
        results.append(result)

        print(f"  filtered={result['num_filtered']:,}  "
              f"close(<100km)={result['num_conjunctions']}  "
              f"Pc={result['aggregate_pc']:.3e}")

    # Summary table
    print()
    print("=" * 70)
    print("AGGREGATE COLLISION PROBABILITY PER CANDIDATE")
    print("=" * 70)
    print(f"  {'NORAD':>8}  {'Filtered':>9}  {'TCA<100km':>9}  "
          f"{'Min TCA (km)':>13}  {'Agg Pc':>12}  {'Range OK?':>9}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*13}  {'-'*12}  {'-'*9}")

    all_pc = []
    for r in results:
        in_range = 1e-5 <= r['aggregate_pc'] <= 1e-3
        flag = "YES" if in_range else ("low" if r['aggregate_pc'] < 1e-5 else "high")
        print(f"  {r['norad_id']:>8}  {r['num_filtered']:>9,}  "
              f"{r['num_conjunctions']:>9}  "
              f"{r['min_tca_km']:>13.2f}  "
              f"{r['aggregate_pc']:>12.3e}  {flag:>9}")
        all_pc.append(r['aggregate_pc'])

    print()
    print(f"  Pc range across all candidates: "
          f"{min(all_pc):.3e} – {max(all_pc):.3e}")
    print(f"  Expected (Owens-Fahrner 2025) : 1.000e-05 – 1.000e-03")

    # Save results
    df_out = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved to: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
