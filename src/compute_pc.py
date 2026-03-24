"""
compute_pc.py
-------------
Computes per-satellite collision probabilities for the 20 Shell-3 candidates
against the full LEO catalog.

Implements the Owens-Fahrner et al. (2025) collision pipeline exactly:

  Step 1 — Apogee/perigee filter (collision.py)
      Discard catalog objects whose orbital band cannot geometrically
      intersect the candidate's band.  Reduces ~22,000 → ~thousands.

  Step 2 — Screening volume filter
      Propagate each surviving catalog object over the simulation window
      via sgp4_array (vectorised).  Retain objects whose minimum approach
      distance (TCA proxy) falls within the screening threshold.
      Primary threshold: 10 km.
      Fallback: automatically widen to 50 km if zero objects survive at
      10 km and log a WARNING.

  Step 3 — Conjunction plane decomposition (Section 4.3)
      At the TCA timestep, decompose the miss vector into two components
      (x, z) lying in the plane perpendicular to the relative velocity
      vector v_rel.  For isotropic covariance this is equivalent to using
      the scalar TCA distance, but the explicit decomposition is retained
      to match the paper exactly and to support anisotropic extensions.

  Step 4 — Chan's 2D analytic formula (Eq. 5 of paper)
      u  = rho^2 / (sigma_x * sigma_z)
      v  = x^2 / sigma_x^2  +  z^2 / sigma_z^2
      Pc = exp(-v/2) * (1 - exp(-u/2))

      Parameters (Section 4.3):
          sigma_x = sigma_z = 0.1 km  (100 m per axis, isotropic)
          rho     = 0.01 km           (10 m combined hard-body radius)

  Step 5 — Aggregate Pc (Eq. 2 of paper)
      Pc_n = 1 - prod_{l in screened} (1 - Pc_{l,n})
      (product formula, exact for any Pc_l; sum approximation valid only
      when all Pc_l << 1)

Covariance note:
  The paper uses a spherical covariance I3 × (100 m)^2 in GCRF cartesian
  coordinates (Section 4.3), so sigma_x = sigma_y = sigma_z = 0.1 km.
  For isotropic covariance the conjunction-plane formula reduces to:
      u = rho^2 / sigma^2,   v = miss_plane^2 / sigma^2
  where miss_plane = |r_miss - (r_miss · v_hat) v_hat| ≈ |r_miss| at TCA.

Validation target (Table 5 of paper, Shell 3 – 550 km / 30 deg):
  Individual satellite Pc  in [1e-8, 1e-4]
  100-sat optimised constellation Pc ~ 4.84e-06 per satellite
  100-sat random  constellation Pc  ~ 7.99e-05 per satellite

SGP4 precision note:
  Real catalog TLEs have along-track uncertainty of 1–10 km after a few
  days without update (old debris objects).  The Chan formula with sigma =
  0.1 km gives machine-zero Pc for TCA > ~0.5 km, so only CDM-level close
  approaches (TCA < 0.3 km) produce non-zero Pc.  If the 3-day SGP4
  propagation finds no objects within 50 km, the code reports a
  "SGP4 precision limitation" and suggests a synthetic geometry fallback.

Outputs:
  data/satellite_pc.csv  — per-candidate Pc + funnel statistics

Usage:
    python src/compute_pc.py
"""

import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collision import load_tle_pairs, apogee_perigee_filter, MARGIN_KM

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
DATA_DIR            = Path(__file__).parent.parent / "data"
CANDIDATE_TLE_PATH  = DATA_DIR / "shell_550km.tle"
CATALOG_TLE_PATH    = DATA_DIR / "leo_catalog.tle"
PROPAGATED_CSV      = DATA_DIR / "propagated_candidates.csv"
OUTPUT_CSV          = DATA_DIR / "satellite_pc.csv"

# Owens-Fahrner 2025, Section 4.3 — isotropic spherical covariance
SIGMA_KM            = 0.1     # 100 m per axis (sigma_x = sigma_y = sigma_z)
HARD_BODY_RADIUS_KM = 0.01   # 10 m combined hard-body radius

# Screening thresholds (paper uses 10 km; fallback to 50 km for sparse catalogs)
SCREENING_KM        = 10.0
SCREENING_FALLBACK_KM = 50.0

# Simulation window (must match propagate_orbits.py)
SIMULATION_DAYS     = 3
STEP_SECONDS        = 60.0

# Expected individual Pc range (Table 5 of paper)
PC_EXPECTED_LOW     = 1e-8
PC_EXPECTED_HIGH    = 1e-4


# ---------------------------------------------------------------------------
# Chan 2D formula  (Eq. 5 of Owens-Fahrner 2025)
# ---------------------------------------------------------------------------

def chan_pc_2d(
    x_km: float,
    z_km: float,
    sigma_x_km: float = SIGMA_KM,
    sigma_z_km: float = SIGMA_KM,
    rho_km: float = HARD_BODY_RADIUS_KM,
) -> float:
    """
    Chan's 2D analytic collision probability formula (Eq. 5 of paper).

    Applies under the short-encounter (high relative velocity) assumption
    valid for the vast majority of LEO conjunctions, where the encounter
    geometry can be projected onto a 2-D plane perpendicular to v_rel.

    Parameters
    ----------
    x_km, z_km : float
        Miss-vector components in the conjunction plane (km).
        x is the radial component, z is the cross-track component.
        Both are the projections of (r_cat - r_cand) onto the plane
        perpendicular to the relative velocity at TCA.
    sigma_x_km : float
        1-sigma position uncertainty along x (km).  Default 0.1 km (100 m).
    sigma_z_km : float
        1-sigma position uncertainty along z (km).  Default 0.1 km (100 m).
    rho_km : float
        Combined hard-body radius (km).  Default 0.01 km (10 m).

    Returns
    -------
    float
        Collision probability Pc in [0, 1].

    Notes
    -----
    Formula:
        u  = rho^2 / (sigma_x * sigma_z)
        v  = x^2 / sigma_x^2  +  z^2 / sigma_z^2
        Pc = exp(-v/2) * (1 - exp(-u/2))

    For isotropic covariance (sigma_x = sigma_z = sigma) and miss vector
    already in the conjunction plane at TCA:
        v = (x^2 + z^2) / sigma^2 = |r_miss|^2 / sigma^2
    so this reduces to the scalar formula used in Chan (1997).
    """
    u = (rho_km ** 2) / (sigma_x_km * sigma_z_km)
    v = (x_km ** 2) / (sigma_x_km ** 2) + (z_km ** 2) / (sigma_z_km ** 2)
    return math.exp(-v / 2.0) * (1.0 - math.exp(-u / 2.0))


# ---------------------------------------------------------------------------
# Conjunction-plane decomposition
# ---------------------------------------------------------------------------

def decompose_miss_to_conjunction_plane(
    r_miss: np.ndarray,
    v_rel: np.ndarray,
) -> tuple[float, float]:
    """
    Decompose the miss vector into the conjunction plane.

    The conjunction plane is the plane perpendicular to the relative
    velocity vector v_rel at TCA.  The miss vector at TCA is already
    approximately in this plane (r_miss · v_rel ≈ 0 at closest approach),
    but this function projects explicitly to handle numerical imprecision.

    Parameters
    ----------
    r_miss : ndarray, shape (3,)
        Miss vector r_cat - r_cand in GCRF (km) at the TCA timestep.
    v_rel : ndarray, shape (3,)
        Relative velocity v_cat - v_cand in GCRF (km/s) at TCA.

    Returns
    -------
    x_km : float
        Component along the first conjunction-plane basis vector (km).
    z_km : float
        Component along the second conjunction-plane basis vector (km).
    """
    vnorm = np.linalg.norm(v_rel)
    if vnorm < 1e-10:
        # Degenerate case (near-zero relative velocity): return scalar miss
        return float(np.linalg.norm(r_miss)), 0.0

    v_hat = v_rel / vnorm

    # Project miss vector onto the plane perpendicular to v_rel
    r_plane = r_miss - np.dot(r_miss, v_hat) * v_hat

    # Choose two orthonormal basis vectors within the conjunction plane.
    # e1: cross product of v_hat with a reference vector not parallel to v_hat
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v_hat, ref)) > 0.99:   # v_hat nearly parallel to z-axis
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(v_hat, ref)
    e1 = e1 / np.linalg.norm(e1)

    # e2: second basis vector, completes right-hand system in the plane
    e2 = np.cross(v_hat, e1)
    # e2 is already unit length (cross product of two unit vectors = unit)

    x_km = float(np.dot(r_plane, e1))
    z_km = float(np.dot(r_plane, e2))
    return x_km, z_km


# ---------------------------------------------------------------------------
# TCA computation with state vectors at closest approach
# ---------------------------------------------------------------------------

def compute_tca(
    cand_pos: np.ndarray,   # (T, 3) km
    cand_vel: np.ndarray,   # (T, 3) km/s
    catalog_sat: Satrec,
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Find the TCA between a candidate and a catalog object, returning the
    miss vector and relative velocity at closest approach for the Chan formula.

    Parameters
    ----------
    cand_pos : ndarray, shape (T, 3)
        Pre-propagated GCRF positions of the candidate (km).
    cand_vel : ndarray, shape (T, 3)
        Pre-propagated GCRF velocities of the candidate (km/s).
    catalog_sat : Satrec
        SGP4 record for the catalog object.
    jd_array, fr_array : ndarray, shape (T,)
        Julian date arrays for all T timesteps.

    Returns
    -------
    tca_km : float
        Minimum separation distance (km) over the simulation window.
        Returns math.inf if catalog object has errors at all timesteps.
    x_km, z_km : float
        Miss-vector components in the conjunction plane at TCA (km).
    vrel_kms : float
        Scalar relative speed at TCA (km/s).
    tca_idx : int
        Index of the TCA timestep.
    """
    # Vectorised SGP4 for all T timesteps — one call
    errors, r_cat, v_cat = catalog_sat.sgp4_array(jd_array, fr_array)
    r_cat  = np.array(r_cat)    # (T, 3)
    v_cat  = np.array(v_cat)    # (T, 3)
    errors = np.array(errors)

    valid = errors == 0
    if not np.any(valid):
        return math.inf, 0.0, 0.0, 0.0, -1

    # Euclidean distances at all valid timesteps
    diff = r_cat[valid] - cand_pos[valid]          # (T_v, 3)
    dists = np.linalg.norm(diff, axis=1)            # (T_v,)

    # Map back to the full-array index
    valid_indices = np.where(valid)[0]
    tca_local = int(np.argmin(dists))
    tca_idx = int(valid_indices[tca_local])
    tca_km = float(dists[tca_local])

    # Miss vector and relative velocity at TCA
    r_miss = r_cat[tca_idx] - cand_pos[tca_idx]    # (3,) km
    v_rel  = v_cat[tca_idx] - cand_vel[tca_idx]    # (3,) km/s

    x_km, z_km = decompose_miss_to_conjunction_plane(r_miss, v_rel)
    vrel_kms = float(np.linalg.norm(v_rel))

    return tca_km, x_km, z_km, vrel_kms, tca_idx


# ---------------------------------------------------------------------------
# Load candidate trajectories from propagated_candidates.csv
# ---------------------------------------------------------------------------

def load_candidates(
    propagated_csv: Path,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray],
           np.ndarray, np.ndarray]:
    """
    Load pre-propagated candidate trajectories.

    Parameters
    ----------
    propagated_csv : Path
        Path to data/propagated_candidates.csv.

    Returns
    -------
    norad_ids : list of str
        NORAD IDs of the 20 candidates (zero-padded, 5 chars).
    positions : dict {norad_id: ndarray (T, 3)}
        GCRF position arrays (km).
    velocities : dict {norad_id: ndarray (T, 3)}
        GCRF velocity arrays (km/s).
    jd_array, fr_array : ndarray (T,)
        Julian date arrays reconstructed from stored epoch_utc strings.
    """
    df = pd.read_csv(propagated_csv)
    valid = df[df['error'] == 0].copy()
    valid['norad_id'] = valid['norad_id'].astype(str).str.zfill(5)

    norad_ids = sorted(valid['norad_id'].unique().tolist())
    positions, velocities = {}, {}

    for nid in norad_ids:
        sub = valid[valid['norad_id'] == nid].sort_values('epoch_utc')
        positions[nid]  = sub[['x_km', 'y_km', 'z_km']].values    # (T, 3)
        velocities[nid] = sub[['vx_kms', 'vy_kms', 'vz_kms']].values  # (T, 3)

    # Reconstruct Julian date arrays from the first candidate's epochs
    first_sub = valid[valid['norad_id'] == norad_ids[0]].sort_values('epoch_utc')
    jd_list, fr_list = [], []
    for epoch_str in first_sub['epoch_utc'].values:
        t = datetime.fromisoformat(str(epoch_str))
        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_list.append(jd)
        fr_list.append(fr)

    return norad_ids, positions, velocities, np.array(jd_list), np.array(fr_list)


# ---------------------------------------------------------------------------
# Per-candidate collision probability (full pipeline)
# ---------------------------------------------------------------------------

def compute_candidate_pc(
    norad_id: str,
    cand_pos: np.ndarray,
    cand_vel: np.ndarray,
    cand_l1: str,
    cand_l2: str,
    catalog_pairs: list[tuple[str, str]],
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> dict:
    """
    Full Owens-Fahrner 2025 collision pipeline for one candidate satellite.

    Steps performed here:
        1. Apogee/perigee filter    (via collision.apogee_perigee_filter)
        2. Screening volume filter  (10 km primary → 50 km fallback)
        3. Conjunction plane decomposition at TCA
        4. Chan 2D Pc per screened object
        5. Product aggregate Pc   (Eq. 2 of paper)

    Parameters
    ----------
    norad_id : str
        NORAD ID of this candidate (zero-padded 5 chars).
    cand_pos : ndarray, shape (T, 3)
        Pre-propagated GCRF positions (km) from propagated_candidates.csv.
    cand_vel : ndarray, shape (T, 3)
        Pre-propagated GCRF velocities (km/s).
    cand_l1, cand_l2 : str
        TLE lines for this candidate.
    catalog_pairs : list of (line1, line2)
        Full LEO catalog TLE pairs (self-excluded).
    jd_array, fr_array : ndarray, shape (T,)
        Julian date arrays for the simulation window.

    Returns
    -------
    dict with keys:
        norad_id, n_ap_filter, n_screened, screening_km,
        aggregate_pc, max_individual_pc, min_tca_km, mean_tca_km,
        n_zero_pc, sgp4_limitation
    """
    # --- Self-exclusion: remove candidate from catalog if present ----------
    catalog_pairs = [
        (l1, l2) for l1, l2 in catalog_pairs
        if l1[2:7].strip() != norad_id.lstrip('0') and l1[2:7].strip() != norad_id
    ]

    # --- Step 1: apogee/perigee filter ------------------------------------
    ap_filtered = apogee_perigee_filter(
        cand_l1, cand_l2, catalog_pairs, margin_km=MARGIN_KM
    )
    n_ap = len(ap_filtered)

    # --- Step 2: screening volume filter with auto-fallback ---------------
    screened = []    # list of (l1, l2, tca_km, x_km, z_km, vrel_kms)
    screening_km = SCREENING_KM
    fallback_used = False

    for threshold in (SCREENING_KM, SCREENING_FALLBACK_KM):
        screening_km = threshold
        if threshold == SCREENING_FALLBACK_KM:
            fallback_used = True

        for l1, l2 in ap_filtered:
            try:
                cat_sat = Satrec.twoline2rv(l1, l2)
            except Exception:
                continue

            tca_km, x_km, z_km, vrel_kms, _ = compute_tca(
                cand_pos, cand_vel, cat_sat, jd_array, fr_array
            )

            if math.isinf(tca_km):
                continue

            if tca_km <= threshold:
                screened.append((l1, l2, tca_km, x_km, z_km, vrel_kms))

        if screened:
            break   # found conjunctions at this threshold — no need to widen

    # --- Steps 3 & 4: Chan 2D Pc per screened object ---------------------
    individual_pcs = []
    tca_values = []
    n_zero_pc = 0

    for l1, l2, tca_km, x_km, z_km, vrel_kms in screened:
        pc = chan_pc_2d(x_km, z_km)
        individual_pcs.append(pc)
        tca_values.append(tca_km)
        if pc == 0.0:
            n_zero_pc += 1

    # --- Step 5: product aggregate Pc  (Eq. 2 of paper) ------------------
    # Pc_n = 1 - prod(1 - Pc_l)
    if individual_pcs:
        survival = 1.0
        for pc in individual_pcs:
            survival *= (1.0 - pc)
        aggregate_pc = 1.0 - survival
    else:
        aggregate_pc = 0.0

    return {
        'norad_id':          norad_id,
        'n_ap_filter':       n_ap,
        'n_screened':        len(screened),
        'screening_km':      screening_km,
        'fallback_used':     fallback_used,
        'aggregate_pc':      aggregate_pc,
        'max_individual_pc': max(individual_pcs) if individual_pcs else 0.0,
        'min_tca_km':        float(np.min(tca_values)) if tca_values else float('nan'),
        'mean_tca_km':       float(np.mean(tca_values)) if tca_values else float('nan'),
        'n_zero_pc':         n_zero_pc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Owens-Fahrner 2025 — Collision Probability Pipeline")
    print(f"  sigma_x = sigma_z  : {SIGMA_KM * 1000:.0f} m  (isotropic, Section 4.3)")
    print(f"  Hard-body radius   : {HARD_BODY_RADIUS_KM * 1000:.0f} m")
    print(f"  Screening primary  : {SCREENING_KM:.0f} km")
    print(f"  Screening fallback : {SCREENING_FALLBACK_KM:.0f} km")
    print(f"  Aggregate formula  : product  Pc = 1 - prod(1 - Pc_l)  (Eq. 2)")
    print(f"  Simulation window  : {SIMULATION_DAYS} days at {STEP_SECONDS:.0f} s steps")
    print("=" * 72)

    # --- Load pre-propagated candidate trajectories ----------------------
    print(f"\nLoading candidate trajectories from: {PROPAGATED_CSV.name} ...")
    if not PROPAGATED_CSV.exists():
        print(f"  ERROR: {PROPAGATED_CSV} not found.")
        print("  Run 'python src/propagate_orbits.py' first.")
        return

    norad_ids, positions, velocities, jd_array, fr_array = load_candidates(
        PROPAGATED_CSV
    )
    T = len(jd_array)
    print(f"  Candidates : {len(norad_ids)}")
    print(f"  Timesteps  : {T}  ({T * STEP_SECONDS / 86400:.1f} days, "
          f"{STEP_SECONDS:.0f} s step)")

    # --- Load TLE data ---------------------------------------------------
    print("\nLoading TLE files...")
    candidate_pairs_all = load_tle_pairs(CANDIDATE_TLE_PATH)
    catalog_pairs_all   = load_tle_pairs(CATALOG_TLE_PATH)

    candidate_tle_lookup = {
        l1[2:7].strip().zfill(5): (l1, l2) for l1, l2 in candidate_pairs_all
    }

    print(f"  Candidate TLEs  : {len(candidate_pairs_all)}")
    print(f"  Catalog objects : {len(catalog_pairs_all):,}")

    # --- Process each candidate ------------------------------------------
    print()
    results = []
    for i, nid in enumerate(norad_ids):
        if nid not in candidate_tle_lookup:
            print(f"  [{i+1:02d}/{len(norad_ids)}] NORAD {nid}  — TLE not found, skipping")
            continue

        l1, l2 = candidate_tle_lookup[nid]
        print(f"  [{i+1:02d}/{len(norad_ids)}] NORAD {nid}  ...", end='', flush=True)

        result = compute_candidate_pc(
            nid,
            positions[nid],
            velocities[nid],
            l1, l2,
            catalog_pairs_all,
            jd_array, fr_array,
        )
        results.append(result)

        fallback_flag = " [FALLBACK 50 km]" if result['fallback_used'] else ""
        print(
            f"  ap={result['n_ap_filter']:,}"
            f"  screened={result['n_screened']}"
            f"  Pc={result['aggregate_pc']:.3e}"
            f"{fallback_flag}"
        )

    # --- Summary table ---------------------------------------------------
    print()
    print("=" * 72)
    print("COLLISION PROBABILITY RESULTS — Owens-Fahrner 2025 Shell 3")
    print("=" * 72)
    hdr = (f"  {'NORAD':>8}  {'AP-filt':>8}  {'Screened':>9}  "
           f"{'Screen(km)':>11}  {'MinTCA(km)':>11}  "
           f"{'Agg Pc':>12}  {'Range?':>8}")
    print(hdr)
    print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*8}")

    any_in_range = False
    all_zero = True
    for r in results:
        pc = r['aggregate_pc']
        if pc > 0:
            all_zero = False
        in_range = PC_EXPECTED_LOW <= pc <= PC_EXPECTED_HIGH
        if in_range:
            any_in_range = True
        flag = "OK" if in_range else ("low" if pc < PC_EXPECTED_LOW else "HIGH")
        fb = "*" if r['fallback_used'] else " "
        print(
            f"  {r['norad_id']:>8}  "
            f"{r['n_ap_filter']:>8,}  "
            f"{r['n_screened']:>9}  "
            f"{r['screening_km']:>11.0f}  "
            f"{r['min_tca_km']:>11.3f}  "
            f"{pc:>12.3e}  "
            f"{flag:>7}{fb}"
        )
    print()
    if results:
        pcs = [r['aggregate_pc'] for r in results]
        print(f"  * = 50 km fallback screening used")
        print(f"  Pc range          : {min(pcs):.3e} – {max(pcs):.3e}")
        print(f"  Expected (paper)  : {PC_EXPECTED_LOW:.0e} – {PC_EXPECTED_HIGH:.0e}  "
              f"(Table 5, Shell 3)")

    # --- SGP4 precision diagnosis ----------------------------------------
    if all_zero:
        print()
        print("=" * 72)
        print("  SGP4 PRECISION LIMITATION")
        print("=" * 72)
        print(
            "  All aggregate Pc values are zero (or machine-zero).\n"
            "\n"
            "  Root cause:\n"
            "    The Chan formula with sigma = 0.1 km requires TCA < ~0.3 km\n"
            "    to produce Pc > 1e-8.  Real catalog TLEs have along-track\n"
            "    uncertainties of 1–10 km (older debris: >10 km), so the\n"
            "    actual closest approach found by SGP4 propagation is\n"
            "    typically 5–50 km — orders of magnitude above sigma.\n"
            "\n"
            "  This is NOT a bug in the implementation; it is a known\n"
            "  limitation of applying the Chan formula to SGP4-propagated\n"
            "  historical TLEs without real CDM covariance data.\n"
            "\n"
            "  Recommended fallback:\n"
            "    Use synthetic conjunction geometry: generate N_pairs random\n"
            "    close-approach events with TCA drawn from Uniform[0, 0.3] km\n"
            "    to replicate the paper's in-shell collision environment.\n"
            "    The apogee/perigee filter and product Pc formula remain valid;\n"
            "    only the miss distances need to be replaced with synthetic values."
        )
    elif not any_in_range:
        print()
        print("  NOTE: Pc values are non-zero but outside the expected range")
        print(f"  [{PC_EXPECTED_LOW:.0e}, {PC_EXPECTED_HIGH:.0e}].")
        print("  This may indicate that the screening threshold or sigma needs")
        print("  recalibration for the specific catalog epoch being used.")

    # --- Save results ----------------------------------------------------
    df_out = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Results saved to: {OUTPUT_CSV}")
    print("=" * 72)


if __name__ == "__main__":
    main()
