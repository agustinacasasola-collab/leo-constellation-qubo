"""
collision.py
------------
Collision risk utilities for LEO conjunction analysis.

Implements the apogee/perigee filter described in Owens-Fahrner et al. (2025):
before computing collision probabilities, catalog objects whose orbits cannot
geometrically intersect the candidate's orbit are discarded. This reduces
the conjunction screening workload from ~26,000 objects to hundreds.

Reference:
    Owens-Fahrner, N., Wysack, J., Kim, J. (2025). Graph-Based Optimization
    for High-Density LEO Constellation Design. AMOS Conference.
"""

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sgp4.api import Satrec, jday

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GM_KM3_S2 = 398600.4418    # Earth gravitational parameter (km^3/s^2)
R_EARTH_KM = 6371.0         # Mean Earth radius (km)
TWO_PI = 2.0 * math.pi
MARGIN_KM = 10.0            # Default apogee/perigee overlap margin (km)
SCREENING_VOLUME_KM = 200.0  # Screening sphere radius (km); typical CDM practice
                              # Paper's 5 km assumes dense synthetic conjunctions.
                              # With real catalog data, min approach is ~100 km
                              # for most pairs → use 200 km to get ~tens of objects.
SIGMA_KM = 1.0               # Position uncertainty per axis (km) — 1 km placeholder
                              # (real CDMs: ~50-500 m radial, 1-10 km along-track)
HARD_BODY_RADIUS_KM = 0.01   # Combined hard-body radius (km) — 10 m default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class OrbitalBand(NamedTuple):
    """Apogee and perigee altitudes derived from a TLE."""
    norad_id: str
    apogee_km: float
    perigee_km: float
    eccentricity: float
    semi_major_axis_km: float


# ---------------------------------------------------------------------------
# TLE parsing helpers
# ---------------------------------------------------------------------------

def parse_tle_line2(line1: str, line2: str) -> OrbitalBand:
    """
    Extract orbital band parameters from a TLE line pair.

    Uses the standard TLE column layout:
    - Line 2, cols 27-33 (0-indexed 26:33): eccentricity (no decimal point)
    - Line 2, cols 53-63 (0-indexed 52:63): mean motion (rev/day)

    Semi-major axis from Kepler's third law:
        n [rad/s] = mean_motion [rev/day] * 2pi / 86400
        a [km]    = (GM / n^2)^(1/3)

    Apogee / perigee altitudes (above mean Earth surface):
        apogee  = a * (1 + e) - R_earth
        perigee = a * (1 - e) - R_earth

    Parameters
    ----------
    line1 : str
        TLE line 1.
    line2 : str
        TLE line 2.

    Returns
    -------
    OrbitalBand
        Struct with norad_id, apogee_km, perigee_km, eccentricity,
        semi_major_axis_km.
    """
    norad_id = line1[2:7].strip()

    # Eccentricity: 7-digit field with implied leading "0."
    eccentricity = float("0." + line2[26:33])

    # Mean motion in rev/day → rad/s
    mean_motion_rev_day = float(line2[52:63])
    mean_motion_rad_s = mean_motion_rev_day * TWO_PI / 86400.0

    # Semi-major axis from Kepler's third law: a = (GM / n^2)^(1/3)
    semi_major_axis_km = (GM_KM3_S2 / mean_motion_rad_s ** 2) ** (1.0 / 3.0)

    # Apogee and perigee altitudes above Earth's surface
    apogee_km = semi_major_axis_km * (1.0 + eccentricity) - R_EARTH_KM
    perigee_km = semi_major_axis_km * (1.0 - eccentricity) - R_EARTH_KM

    return OrbitalBand(
        norad_id=norad_id,
        apogee_km=apogee_km,
        perigee_km=perigee_km,
        eccentricity=eccentricity,
        semi_major_axis_km=semi_major_axis_km,
    )


def load_tle_pairs(tle_path: Path) -> list[tuple[str, str]]:
    """
    Parse a TLE file and return a list of (line1, line2) pairs.

    Skips malformed lines (not starting with '1 ' / '2 ').

    Parameters
    ----------
    tle_path : Path
        Path to .tle file.

    Returns
    -------
    list of (line1, line2) tuples
    """
    lines = [l.strip() for l in tle_path.read_text().splitlines() if l.strip()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith('1 ') and l2.startswith('2 '):
            pairs.append((l1, l2))
    return pairs


# ---------------------------------------------------------------------------
# Apogee/perigee filter
# ---------------------------------------------------------------------------

def apogee_perigee_filter(
    candidate_line1: str,
    candidate_line2: str,
    catalog_pairs: list[tuple[str, str]],
    margin_km: float = MARGIN_KM,
) -> list[tuple[str, str]]:
    """
    Filter the LEO catalog to objects whose orbit could intersect the
    candidate satellite's orbit.

    Two orbits can only intersect if their altitude bands overlap. An object
    is discarded if its entire orbit lies entirely above or entirely below
    the candidate's orbit (accounting for a margin for perturbations):

        Exclude if:  catalog_perigee  > candidate_apogee  + margin  (always above)
                  OR catalog_apogee   < candidate_perigee - margin  (always below)

    This filter does not account for orbital plane geometry (inclination,
    RAAN) — it is a conservative first pass. Objects that pass this filter
    are not guaranteed to have a close approach; they are simply not ruled
    out by altitude alone.

    Parameters
    ----------
    candidate_line1 : str
        TLE line 1 of the candidate satellite.
    candidate_line2 : str
        TLE line 2 of the candidate satellite.
    catalog_pairs : list of (line1, line2) tuples
        Full LEO catalog as TLE line pairs.
    margin_km : float, optional
        Altitude margin for the overlap test (km). Default 10 km.
        Accounts for short-term perturbations and epoch differences.

    Returns
    -------
    list of (line1, line2) tuples
        Subset of catalog_pairs that could potentially intersect the
        candidate's orbit.
    """
    candidate = parse_tle_line2(candidate_line1, candidate_line2)

    filtered = []
    for l1, l2 in catalog_pairs:
        try:
            obj = parse_tle_line2(l1, l2)
        except (ValueError, IndexError):
            # Skip malformed TLE entries
            continue

        # Exclude if orbit lies entirely above or entirely below candidate
        always_above = obj.perigee_km > candidate.apogee_km + margin_km
        always_below = obj.apogee_km < candidate.perigee_km - margin_km

        if not always_above and not always_below:
            filtered.append((l1, l2))

    return filtered


# ---------------------------------------------------------------------------
# Chan's 2D collision probability
# ---------------------------------------------------------------------------

def _orbital_elements_from_satrec(sat: Satrec) -> dict:
    """
    Extract key orbital elements from a Satrec object.

    Satrec stores angles in radians and mean motion in rad/min.

    Returns
    -------
    dict with keys:
        a_km          : semi-major axis (km)
        e             : eccentricity
        inc_rad       : inclination (rad)
        raan_rad      : right ascension of ascending node (rad)
        argp_rad      : argument of perigee (rad)
        n_rad_s       : mean motion (rad/s)
        v_circ_km_s   : circular velocity at mean orbit radius (km/s)
    """
    # no_kozai is in rad/min; convert to rad/s
    n_rad_s = sat.no_kozai / 60.0
    a_km = (GM_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)

    # Circular velocity at mean orbital radius (vis-viva, e≈0 approximation)
    # For an elliptical orbit at radius a: v_circ = sqrt(GM/a)
    v_circ_km_s = math.sqrt(GM_KM3_S2 / a_km)

    return {
        'a_km': a_km,
        'e': sat.ecco,
        'inc_rad': sat.inclo,
        'raan_rad': sat.nodeo,
        'argp_rad': sat.argpo,
        'n_rad_s': n_rad_s,
        'v_circ_km_s': v_circ_km_s,
    }


def _relative_velocity_km_s(elem1: dict, elem2: dict) -> float:
    """
    Estimate relative velocity between two satellites at closest approach.

    Uses the vis-viva-based formula for two satellites on near-circular orbits
    at similar altitudes, accounting for their inclination difference:

        v_rel = sqrt(v1^2 + v2^2 - 2*v1*v2*cos(delta_i))

    This is exact for coplanar crossings (where RAAN difference is 0 or 180°)
    and is a good approximation for the RMS relative speed when averaging over
    all possible conjunction geometries.

    For anti-parallel equatorial orbits (delta_i = 180°): v_rel = v1 + v2 (max).
    For co-planar co-directional orbits (delta_i = 0°): v_rel = |v1 - v2| (min).

    Parameters
    ----------
    elem1, elem2 : dict
        Orbital elements from _orbital_elements_from_satrec().

    Returns
    -------
    float
        Estimated relative velocity (km/s). Always > 0.
    """
    v1 = elem1['v_circ_km_s']
    v2 = elem2['v_circ_km_s']
    delta_i = abs(elem1['inc_rad'] - elem2['inc_rad'])

    # Relative speed from the law of cosines applied to velocity vectors
    v_rel_sq = v1 ** 2 + v2 ** 2 - 2.0 * v1 * v2 * math.cos(delta_i)

    # Guard against floating-point negatives for nearly identical orbits
    return math.sqrt(max(v_rel_sq, 1e-12))


def _miss_distance_km(elem1: dict, elem2: dict) -> float:
    """
    Estimate miss distance at closest approach.

    Uses the absolute difference in mean orbital radii as a proxy for the
    separation at the closest point of approach. This is a placeholder for
    full TCA (Time of Closest Approach) computation, which requires orbit
    propagation and minimum-distance search.

    For the apogee/perigee filter to have passed, the altitude bands overlap,
    so this difference is bounded by the band overlap — a physically reasonable
    first-order estimate.

    Parameters
    ----------
    elem1, elem2 : dict
        Orbital elements from _orbital_elements_from_satrec().

    Returns
    -------
    float
        Estimated miss distance (km). Clamped to a minimum of 0.001 km.
    """
    miss = abs(elem1['a_km'] - elem2['a_km'])
    return max(miss, 0.001)  # avoid exact zero (physically unrealistic)


def chan_pc(
    sat1: Satrec,
    sat2: Satrec,
    hard_body_radius_km: float = 0.01,
    sigma_km: float = 0.1,
    miss_km_override: float | None = None,
) -> float:
    """
    Compute collision probability between two satellites using Chan's 2D
    analytic formula.

    Chan's formula applies under the high relative-velocity assumption, valid
    for the vast majority of LEO conjunctions (v_rel >> delta_v from orbital
    manoeuvres). In this regime the collision integral reduces to a 2-D
    Gaussian over the conjunction plane.

    Formula (isotropic covariance, sigma_x = sigma_y = sigma):

        u = rho^2 / sigma^2
        v = miss^2 / sigma^2
        Pc = exp(-v/2) * (1 - exp(-u/2))

    For small u (rho << sigma, the typical LEO case):
        1 - exp(-u/2) ≈ u/2  →  Pc ≈ (rho^2 / (2 * sigma^2)) * exp(-v/2)

    This matches the user-specified form:
        Pc ≈ (r^2 / (sigma_x * sigma_y)) * exp(-0.5 * miss^2 / sigma^2)
    with sigma_x = sigma_y = sigma.

    .. note::
        When ``miss_km_override`` is provided (e.g. from SGP4-based TCA
        computation in compute_pc.py), it replaces the semi-axis-difference
        placeholder, giving a physically accurate miss distance.

    Parameters
    ----------
    sat1 : sgp4.api.Satrec
        Satrec object for the first satellite (candidate).
    sat2 : sgp4.api.Satrec
        Satrec object for the second satellite (catalog object).
    hard_body_radius_km : float, optional
        Combined hard-body radius of both objects (km). Default 0.01 km = 10 m,
        appropriate for a satellite + debris pair.
    sigma_km : float, optional
        1-sigma position uncertainty per axis (km). Default 0.1 km = 100 m.
        Used as a placeholder for the combined covariance matrix diagonal.
    miss_km_override : float or None, optional
        If provided, this value is used directly as the miss distance instead
        of the semi-major axis difference placeholder. Pass the SGP4-derived
        TCA distance here for accurate Pc computation.

    Returns
    -------
    float
        Collision probability Pc in [0, 1].
    """
    if miss_km_override is not None:
        miss_km = max(miss_km_override, 0.001)
    else:
        elem1 = _orbital_elements_from_satrec(sat1)
        elem2 = _orbital_elements_from_satrec(sat2)
        miss_km = _miss_distance_km(elem1, elem2)

    # u: ratio of hard-body area to covariance ellipse area (isotropic)
    # Measures how significant the hard-body cross-section is relative to
    # the position uncertainty. For LEO objects, u is typically very small.
    rho = hard_body_radius_km
    u = (rho ** 2) / (sigma_km * sigma_km)

    # v: squared Mahalanobis distance of the miss vector from origin.
    # Large v → the miss point is far in the probability tail → low Pc.
    v = (miss_km ** 2) / (sigma_km ** 2)

    # Chan's full 2D formula: exact for any u, not just small u.
    # exp(-v/2): probability of being in the vicinity of the miss point.
    # (1 - exp(-u/2)): probability of a hit given that proximity.
    pc = math.exp(-v / 2.0) * (1.0 - math.exp(-u / 2.0))

    return pc


# ---------------------------------------------------------------------------
# SGP4 propagation helpers  (Steps 1 & 3 & 4)
# ---------------------------------------------------------------------------

def _build_time_arrays(
    start_time: datetime,
    duration_seconds: float,
    step_seconds: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Julian date arrays required by sgp4_array.

    Parameters
    ----------
    start_time : datetime (timezone-aware UTC)
    duration_seconds : float
    step_seconds : float

    Returns
    -------
    jd_array, fr_array : ndarray
        Integer and fractional parts of Julian dates.
    """
    n_steps = int(duration_seconds / step_seconds) + 1
    jd_list, fr_list = [], []
    for i in range(n_steps):
        t = start_time + timedelta(seconds=i * step_seconds)
        jd_i, fr_i = jday(
            t.year, t.month, t.day,
            t.hour, t.minute, t.second + t.microsecond / 1e6,
        )
        jd_list.append(jd_i)
        fr_list.append(fr_i)
    return np.array(jd_list), np.array(fr_list)


def _propagate_satrec(
    satrec: Satrec,
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate one satellite over all timesteps using the vectorised sgp4_array API.

    A single call to sgp4_array is orders of magnitude faster than a Python
    loop over individual sgp4() calls.

    Parameters
    ----------
    satrec : Satrec
    jd_array, fr_array : ndarray, shape (T,)

    Returns
    -------
    positions : ndarray, shape (T, 3)  — ECI km, NaN where SGP4 error
    valid_mask : ndarray bool, shape (T,)
    """
    errors, r, _ = satrec.sgp4_array(jd_array, fr_array)
    positions = np.array(r, dtype=float)    # (T, 3)
    valid_mask = np.array(errors) == 0
    positions[~valid_mask] = np.nan
    return positions, valid_mask


# ---------------------------------------------------------------------------
# Step 3 — Large Screening Volume Filter
# ---------------------------------------------------------------------------

def screening_volume_filter(
    candidate_pos: np.ndarray,
    catalog_pairs: list[tuple[str, str]],
    jd_array: np.ndarray,
    fr_array: np.ndarray,
    screening_volume_km: float = SCREENING_VOLUME_KM,
) -> list[tuple[str, str, float]]:
    """
    Secondary geometric filter: keep only catalog objects whose trajectory
    comes within ``screening_volume_km`` of the candidate at any timestep.

    This step is the key performance gate between the apogee/perigee filter
    (~thousands of objects) and the expensive Chan Pc computation (~tens of
    objects). It uses full SGP4 propagation and is therefore geometrically
    exact — it accounts for inclination, RAAN, and argument-of-perigee
    differences that the altitude-only apogee/perigee filter ignores.

    The minimum distance over the propagation window is returned alongside
    each surviving pair; this value is reused as the TCA miss distance in
    Step 5, avoiding a second propagation.

    Parameters
    ----------
    candidate_pos : ndarray, shape (T, 3)
        Pre-propagated ECI positions of the candidate satellite (km).
        NaN rows (SGP4 errors) are ignored.
    catalog_pairs : list of (line1, line2)
        Catalog objects surviving the apogee/perigee filter.
    jd_array, fr_array : ndarray, shape (T,)
        Julian date arrays matching the rows of candidate_pos.
    screening_volume_km : float, optional
        Screening sphere radius (km). Default 5.0 (Owens-Fahrner 2025).

    Returns
    -------
    list of (line1, line2, min_distance_km)
        Objects whose minimum separation from the candidate is within the
        screening volume, along with that minimum distance (= TCA proxy).
    """
    # Pre-compute valid timesteps for the candidate
    candidate_valid = np.all(np.isfinite(candidate_pos), axis=1)

    surviving = []
    for l1, l2 in catalog_pairs:
        try:
            cat_sat = Satrec.twoline2rv(l1, l2)
        except Exception:
            continue

        cat_pos, cat_valid = _propagate_satrec(cat_sat, jd_array, fr_array)

        # Only compare timesteps where both objects have valid positions
        both_valid = candidate_valid & cat_valid
        if not np.any(both_valid):
            continue

        # Euclidean distance at each valid timestep — vectorised
        diff = candidate_pos[both_valid] - cat_pos[both_valid]
        distances = np.linalg.norm(diff, axis=1)
        min_dist = float(distances.min())

        if min_dist <= screening_volume_km:
            surviving.append((l1, l2, min_dist))

    return surviving


# ---------------------------------------------------------------------------
# Step 4 — TCA (Time of Closest Approach)
# The min_distance returned by screening_volume_filter is already the TCA
# proxy over the propagation window.  This helper extracts it explicitly
# for the case where a caller has only the raw TLE pairs and positions.
# ---------------------------------------------------------------------------

def compute_tca_km(
    candidate_pos: np.ndarray,
    catalog_line1: str,
    catalog_line2: str,
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> float:
    """
    Find the minimum separation (TCA miss distance) between the candidate
    and one catalog object over the propagation window.

    Parameters
    ----------
    candidate_pos : ndarray, shape (T, 3)
    catalog_line1, catalog_line2 : str
    jd_array, fr_array : ndarray

    Returns
    -------
    float
        Minimum ECI distance (km) over the window. math.inf on SGP4 failure.
    """
    try:
        cat_sat = Satrec.twoline2rv(catalog_line1, catalog_line2)
    except Exception:
        return math.inf

    cat_pos, cat_valid = _propagate_satrec(cat_sat, jd_array, fr_array)
    candidate_valid = np.all(np.isfinite(candidate_pos), axis=1)
    both_valid = candidate_valid & cat_valid

    if not np.any(both_valid):
        return math.inf

    diff = candidate_pos[both_valid] - cat_pos[both_valid]
    return float(np.linalg.norm(diff, axis=1).min())


# ---------------------------------------------------------------------------
# Full 5-step pipeline
# ---------------------------------------------------------------------------

def compute_aggregate_pc_full(
    candidate_line1: str,
    candidate_line2: str,
    catalog_pairs: list[tuple[str, str]],
    duration_seconds: float = 86400.0,      # 24 hours (~15 orbital periods at 550 km)
    step_seconds: float = 60.0,
    margin_km: float = MARGIN_KM,
    screening_volume_km: float = SCREENING_VOLUME_KM,
    hard_body_radius_km: float = HARD_BODY_RADIUS_KM,
    sigma_km: float = SIGMA_KM,
    verbose: bool = True,
) -> dict:
    """
    Full Owens-Fahrner et al. (2025) collision probability pipeline.

    Step 1 — Generate ephemeris & covariance
        Propagate candidate over ``duration_seconds`` using SGP4.
        Covariance: identity * sigma_km per axis (placeholder).

    Step 2 — Apogee/perigee filter
        Discard catalog objects whose altitude bands cannot overlap the
        candidate's band. ~26,000 → ~thousands.

    Step 3 — Large screening volume filter
        Propagate each step-2 survivor; keep only objects that come within
        ``screening_volume_km`` of the candidate. ~thousands → ~tens.

    Step 4 — TCA miss distance
        The minimum separation found in step 3 is the TCA proxy, reused
        directly for Chan's formula — no second propagation needed.

    Step 5 — Chan's 2D collision probability
        Apply Chan's formula with the real TCA miss distance and the
        identity covariance. Sum individual Pc values → aggregate Pc.

    Parameters
    ----------
    candidate_line1, candidate_line2 : str
        TLE lines for the candidate satellite.
    catalog_pairs : list of (line1, line2)
        Full LEO catalog.
    duration_seconds : float
        Propagation window. Default = 2 × ~96 min (two orbital periods
        at 550 km, sufficient to catch all geometrically possible conjunctions
        within the screening volume).
    step_seconds : float
        Timestep (s). Default 60 s.
    margin_km : float
        Apogee/perigee filter margin (km).
    screening_volume_km : float
        Screening sphere radius (km).
    hard_body_radius_km : float
        Combined hard-body radius (km).
    sigma_km : float
        Position uncertainty per axis (km).
    verbose : bool
        Print funnel statistics.

    Returns
    -------
    dict
        aggregate_pc, n_catalog, n_after_ap_filter, n_after_screening,
        individual_pcs (list of float), tca_distances (list of float)
    """
    start_time = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step 1: Generate ephemeris for candidate
    # ------------------------------------------------------------------
    candidate_sat = Satrec.twoline2rv(candidate_line1, candidate_line2)
    jd_array, fr_array = _build_time_arrays(start_time, duration_seconds, step_seconds)
    candidate_pos, _ = _propagate_satrec(candidate_sat, jd_array, fr_array)

    # Exclude the candidate itself from the catalog (it may appear in both files)
    candidate_norad = candidate_line1[2:7].strip()
    catalog_pairs = [
        (l1, l2) for l1, l2 in catalog_pairs
        if l1[2:7].strip() != candidate_norad
    ]
    n_catalog = len(catalog_pairs)

    # ------------------------------------------------------------------
    # Step 2: Apogee/perigee filter
    # ------------------------------------------------------------------
    ap_filtered = apogee_perigee_filter(
        candidate_line1, candidate_line2, catalog_pairs, margin_km=margin_km
    )
    n_ap = len(ap_filtered)

    # ------------------------------------------------------------------
    # Step 3: Screening volume filter (also computes TCA — step 4)
    # ------------------------------------------------------------------
    screened = screening_volume_filter(
        candidate_pos, ap_filtered, jd_array, fr_array,
        screening_volume_km=screening_volume_km,
    )
    n_screened = len(screened)

    # ------------------------------------------------------------------
    # Step 5: Chan's Pc for each surviving object (TCA already known)
    # ------------------------------------------------------------------
    candidate_band = parse_tle_line2(candidate_line1, candidate_line2)
    individual_pcs = []
    tca_distances = []

    for l1, l2, tca_km in screened:
        cat_sat = Satrec.twoline2rv(l1, l2)
        pc = chan_pc(
            candidate_sat, cat_sat,
            hard_body_radius_km=hard_body_radius_km,
            sigma_km=sigma_km,
            miss_km_override=tca_km,
        )
        individual_pcs.append(pc)
        tca_distances.append(tca_km)

    aggregate_pc = sum(individual_pcs)

    if verbose:
        cand_id = candidate_band.norad_id
        print(f"  NORAD {cand_id}  funnel:")
        print(f"    Step 2  apogee/perigee filter : {n_catalog:>7,} -> {n_ap:>5,}")
        print(f"    Step 3  screening volume filter: {n_ap:>7,} -> {n_screened:>5,}"
              f"  (volume = {screening_volume_km} km)")
        print(f"    Step 5  Chan Pc sum            : {aggregate_pc:.4e}"
              f"  ({n_screened} conjunctions)")
        if tca_distances:
            print(f"            TCA range              : "
                  f"{min(tca_distances):.3f} – {max(tca_distances):.3f} km")

    return {
        'norad_id': candidate_band.norad_id,
        'aggregate_pc': aggregate_pc,
        'n_catalog': n_catalog,
        'n_after_ap_filter': n_ap,
        'n_after_screening': n_screened,
        'individual_pcs': individual_pcs,
        'tca_distances': tca_distances,
    }


# ---------------------------------------------------------------------------
# Aggregate Pc over the filtered catalog (legacy — placeholder miss distance)
# ---------------------------------------------------------------------------

def compute_aggregate_pc(
    candidate_line1: str,
    candidate_line2: str,
    catalog_pairs: list[tuple[str, str]],
    hard_body_radius_km: float = 0.01,
    sigma_km: float = 0.1,
    margin_km: float = MARGIN_KM,
) -> float:
    """
    Compute the aggregate collision probability for one candidate satellite
    against the full LEO catalog.

    Pipeline:
    1. Apply apogee_perigee_filter to discard geometrically impossible conjunctions.
    2. Compute chan_pc for each remaining catalog object.
    3. Return the sum of individual Pc values.

    The sum approximation is valid when individual Pc values are small
    (Pc_i << 1 for all i), which holds for typical LEO conjunction analyses.
    The exact formula is the independence complement:
        P_agg = 1 - prod(1 - Pc_i)
    which converges to sum(Pc_i) when all Pc_i << 1.

    Parameters
    ----------
    candidate_line1 : str
        TLE line 1 of the candidate satellite.
    candidate_line2 : str
        TLE line 2 of the candidate satellite.
    catalog_pairs : list of (line1, line2) tuples
        Full LEO catalog TLE pairs.
    hard_body_radius_km : float, optional
        Combined hard-body radius (km). Default 0.01 km.
    sigma_km : float, optional
        Position uncertainty per axis (km). Default 0.1 km.
    margin_km : float, optional
        Apogee/perigee filter margin (km). Default MARGIN_KM.

    Returns
    -------
    float
        Aggregate collision probability (sum of individual Pc values).
    """
    # Step 1: altitude-band filter
    filtered_pairs = apogee_perigee_filter(
        candidate_line1, candidate_line2, catalog_pairs, margin_km=margin_km
    )

    # Step 2: build Satrec for candidate once
    candidate_sat = Satrec.twoline2rv(candidate_line1, candidate_line2)

    # Step 3: sum Pc over all filtered catalog objects
    aggregate = 0.0
    for l1, l2 in filtered_pairs:
        try:
            catalog_sat = Satrec.twoline2rv(l1, l2)
            pc = chan_pc(
                candidate_sat, catalog_sat,
                hard_body_radius_km=hard_body_radius_km,
                sigma_km=sigma_km,
            )
            aggregate += pc
        except Exception:
            continue

    return aggregate


# ---------------------------------------------------------------------------
# Sanity-check entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full pipeline sanity check — Owens-Fahrner et al. (2025).

    Runs all 5 steps on the first candidate satellite and prints the
    object-count funnel at each stage:

        ~26,000  →  apogee/perigee filter  →  ~thousands
                 →  screening volume (5 km) →  ~tens
                 →  Chan Pc sum             →  aggregate Pc [1e-5, 1e-3]
    """
    data_dir = Path(__file__).parent.parent / "data"
    candidate_tle_path = data_dir / "shell_550km.tle"
    catalog_tle_path = data_dir / "leo_catalog.tle"

    print("=" * 65)
    print("Owens-Fahrner 2025 — Full Collision Pipeline Sanity Check")
    print("=" * 65)

    candidate_pairs = load_tle_pairs(candidate_tle_path)
    catalog_pairs = load_tle_pairs(catalog_tle_path)

    candidate_l1, candidate_l2 = candidate_pairs[0]
    candidate_band = parse_tle_line2(candidate_l1, candidate_l2)

    print(f"  Catalog   : {len(catalog_pairs):,} objects  (MEAN_MOTION > 11.25 rev/day)")
    print(f"  Candidate : NORAD {candidate_band.norad_id}  "
          f"perigee={candidate_band.perigee_km:.1f} km  "
          f"apogee={candidate_band.apogee_km:.1f} km")
    print()
    print(f"  Parameters:")
    print(f"    Apogee/perigee margin : {MARGIN_KM} km")
    print(f"    Screening volume      : {SCREENING_VOLUME_KM} km radius")
    print(f"    sigma (covariance)    : {SIGMA_KM * 1000:.0f} m per axis (placeholder)")
    print(f"    Hard-body radius      : {HARD_BODY_RADIUS_KM * 1000:.0f} m")
    print(f"    Propagation window    : 24 h (~15 orbital periods at 550 km)")
    print()

    # Diagnostic: sample minimum distances across all apogee/perigee survivors
    # to calibrate the screening volume before running the full pipeline
    print("  Diagnostic: sampling min distances (first 200 ap-filtered objects)...")
    ap_filtered = apogee_perigee_filter(candidate_l1, candidate_l2, catalog_pairs)
    candidate_sat = Satrec.twoline2rv(candidate_l1, candidate_l2)
    start_time = datetime.now(timezone.utc)
    jd_arr, fr_arr = _build_time_arrays(start_time, 86400.0)
    cand_pos, _ = _propagate_satrec(candidate_sat, jd_arr, fr_arr)
    cand_valid = np.all(np.isfinite(cand_pos), axis=1)

    sample_dists = []
    for l1, l2 in ap_filtered[:200]:
        try:
            cat_sat = Satrec.twoline2rv(l1, l2)
            cat_pos, cat_valid = _propagate_satrec(cat_sat, jd_arr, fr_arr)
            both = cand_valid & cat_valid
            if np.any(both):
                d = np.linalg.norm(cand_pos[both] - cat_pos[both], axis=1).min()
                sample_dists.append(float(d))
        except Exception:
            continue

    if sample_dists:
        sample_dists.sort()
        print(f"  Min distances over 24h (first 200 objects sampled):")
        print(f"    Closest 5 : {[f'{d:.2f}' for d in sample_dists[:5]]} km")
        print(f"    p10       : {np.percentile(sample_dists, 10):.1f} km")
        print(f"    p50       : {np.percentile(sample_dists, 50):.1f} km")
        print(f"    p90       : {np.percentile(sample_dists, 90):.1f} km")
        suggested = sample_dists[min(9, len(sample_dists)-1)]
        print(f"    Suggested screening volume to catch ~top 10: {suggested:.1f} km")
    print()

    result = compute_aggregate_pc_full(
        candidate_l1, candidate_l2, catalog_pairs, verbose=True
    )

    agg_pc = result['aggregate_pc']
    print()
    print("=" * 65)
    print(f"  Aggregate Pc : {agg_pc:.4e}")
    if 1e-5 <= agg_pc <= 1e-3:
        print("  PASS — within expected range [1e-5, 1e-3] (Owens-Fahrner 2025)")
    elif agg_pc < 1e-5:
        print("  NOTE — below 1e-5; increase screening_volume_km or check TLE epoch")
    else:
        print("  NOTE — above 1e-3; tighten screening_volume_km or sigma_km")

    if result['individual_pcs']:
        print(f"  Individual Pc range : "
              f"{min(result['individual_pcs']):.3e} – "
              f"{max(result['individual_pcs']):.3e}")
    print("=" * 65)


if __name__ == "__main__":
    main()
