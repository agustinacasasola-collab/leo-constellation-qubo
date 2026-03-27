"""
verify_separation.py
--------------------
Verifies the minimum angular separation between all satellite pairs in the
Arnas (2021) Shell 3 synthetic constellation (data/shell3_synthetic.tle)
using the analytical formula from Arnas (2021) Equation 7.

Mathematical derivation
-----------------------
For two satellites in circular orbits at the same altitude, let alpha be
the argument of latitude of satellite 1 at some epoch.  Then satellite 2 is
at alpha + dM (same mean motion, constant relative phase).  Their 3-D
position unit vectors in ECI (setting Omega_1 = 0 WLOG) are:

    r1 = [cos(alpha),  cos(i)*sin(alpha),  sin(i)*sin(alpha)]
    r2 = [cos(dOmega)*cos(alpha+dM) - sin(dOmega)*cos(i2)*sin(alpha+dM),
          sin(dOmega)*cos(alpha+dM) + cos(dOmega)*cos(i2)*sin(alpha+dM),
          sin(i2)*sin(alpha+dM)]

The dot product cos(theta) = r1 . r2 can be written as a quadratic form
in [cos(alpha), sin(alpha)]:

    cos(theta(alpha)) = A*cos^2(alpha) + (B+C)*cos(alpha)*sin(alpha)
                        + D*sin^2(alpha)

where (Arnas 2021, Eq. 7):

    A = cos(dOmega)*cos(dM) - sin(dOmega)*cos(i1)*sin(dM)
    B = -cos(dOmega)*sin(dM) - sin(dOmega)*cos(i1)*cos(dM)
    C = cos(i2)*sin(dOmega)*cos(dM) + cos(i1)*cos(i2)*cos(dOmega)*sin(dM)
        + sin(i1)*sin(i2)*sin(dM)
    D = -cos(i2)*sin(dOmega)*sin(dM) + cos(i1)*cos(i2)*cos(dOmega)*cos(dM)
        + sin(i1)*sin(i2)*cos(dM)

The maximum of this quadratic form (= cos of the MINIMUM angular separation)
is the largest eigenvalue of the symmetric matrix
    [[A, (B+C)/2], [(B+C)/2, D]]:

    cos(amin) = 0.5 * (A + D + sqrt((A-D)^2 + (B+C)^2))
    amin      = arccos(cos(amin))

Uniformity property of 2D-LFCs (Arnas 2021, Thm 1)
----------------------------------------------------
For a Lattice Flower Constellation the set of angular difference pairs
{(dOmega_1j, dM_1j) : j != 1} equals the set for any other reference
satellite.  Therefore the global minimum separation is attained among the
N-1 pairs involving satellite 1, and a full O(N^2) sweep is unnecessary.

Usage
-----
    python src/verify_separation.py
"""

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and parameters
# ---------------------------------------------------------------------------
TLE_PATH  = Path(__file__).parent.parent / "data" / "shell3_synthetic.tle"
INC_DEG   = 30.0                  # inclination of all Shell 3 satellites
INC_RAD   = math.radians(INC_DEG)
MIN_SEP_THRESHOLD_DEG = 1.0       # Arnas guarantee lower bound

# Pre-compute shared trig values (same inclination for all satellites)
COS_I = math.cos(INC_RAD)
SIN_I = math.sin(INC_RAD)


# ---------------------------------------------------------------------------
# TLE parser (line-2 fields only)
# ---------------------------------------------------------------------------

def parse_tle_file(path: Path) -> list[tuple[int, float, float]]:
    """
    Parse a two-line element file and return orbital elements for each object.

    Reads only Line 2 (RAAN and mean anomaly).  Does NOT call SGP4.

    Returns
    -------
    list of (norad_id, raan_deg, mean_anomaly_deg)
        All angles in degrees, range [0, 360).
    """
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]
    satellites = []
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if not (l1.startswith('1 ') and l2.startswith('2 ')):
            continue
        norad_id = int(l1[2:7])
        # Line 2 column layout (0-indexed):
        #   [17:25] RAAN (deg)
        #   [43:51] mean anomaly (deg)
        raan = float(l2[17:25]) % 360.0
        m    = float(l2[43:51]) % 360.0
        satellites.append((norad_id, raan, m))
    return satellites


# ---------------------------------------------------------------------------
# Arnas (2021) Equation 7 — minimum angular separation
# ---------------------------------------------------------------------------

def cos_min_separation(
    d_omega_rad: float,
    d_m_rad: float,
    cos_i1: float = COS_I,
    sin_i1: float = SIN_I,
    cos_i2: float = COS_I,
    sin_i2: float = SIN_I,
) -> float:
    """
    Compute cos(amin) between two circular co-altitude satellites via
    Arnas (2021) Equation 7.

    Parameters
    ----------
    d_omega_rad : float
        Difference in RAAN (Omega_2 - Omega_1), radians.
    d_m_rad : float
        Difference in mean anomaly (M_2 - M_1), radians.
    cos_i1, sin_i1 : float
        cos/sin of inclination of satellite 1.
    cos_i2, sin_i2 : float
        cos/sin of inclination of satellite 2.

    Returns
    -------
    float
        cos(amin) in [-1, 1].  Take arccos to get the angle.
    """
    cdO = math.cos(d_omega_rad)
    sdO = math.sin(d_omega_rad)
    cdM = math.cos(d_m_rad)
    sdM = math.sin(d_m_rad)

    A = cdO * cdM   - sdO * cos_i1 * sdM
    B = -cdO * sdM  - sdO * cos_i1 * cdM
    C = (cos_i2 * sdO * cdM
         + cos_i1 * cos_i2 * cdO * sdM
         + sin_i1 * sin_i2 * sdM)
    D = (-cos_i2 * sdO * sdM
         + cos_i1 * cos_i2 * cdO * cdM
         + sin_i1 * sin_i2 * cdM)

    # Maximum eigenvalue of [[A, (B+C)/2], [(B+C)/2, D]]
    cos_a = 0.5 * (A + D + math.sqrt((A - D) ** 2 + (B + C) ** 2))

    # Clamp for numerical safety (floating-point rounding near ±1)
    return max(-1.0, min(1.0, cos_a))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Arnas (2021) Eq. 7 — Minimum Angular Separation Verification")
    print(f"  TLE file    : {TLE_PATH.name}")
    print(f"  Inclination : {INC_DEG:.1f} deg (all satellites, i1 = i2)")
    print(f"  Pass threshold : >= {MIN_SEP_THRESHOLD_DEG:.1f} deg")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load TLE file
    # ------------------------------------------------------------------
    if not TLE_PATH.exists():
        print(f"\n  ERROR: {TLE_PATH} not found.")
        print("  Run 'python src/generate_candidates.py' first.")
        return

    sats = parse_tle_file(TLE_PATH)
    N = len(sats)
    print(f"\n  Loaded {N} satellites  (NORAD {sats[0][0]} .. {sats[-1][0]})")

    # ------------------------------------------------------------------
    # 2D-LFC uniformity: check satellite 0 against all others only
    #
    # By Arnas (2021) Thm 1, the set of pairwise angular-difference vectors
    # {(dOmega_0j, dM_0j) : j = 1..N-1} is a permutation of the complete
    # set of pairwise differences in the constellation.  Therefore the
    # minimum over these N-1 pairs equals the global constellation minimum.
    # ------------------------------------------------------------------
    ref_norad, ref_raan, ref_m = sats[0]
    print(f"\n  Reference satellite : NORAD {ref_norad}"
          f"  RAAN = {ref_raan:.4f} deg   M = {ref_m:.4f} deg")
    print(f"  Pairs evaluated     : {N - 1:,}  (satellite 0 vs all others)")

    global_min_sep  = math.inf
    closest_norad   = None
    closest_dOmega  = None
    closest_dM      = None

    for norad_j, raan_j, m_j in sats[1:]:
        d_omega = math.radians(raan_j - ref_raan)
        d_m     = math.radians(m_j    - ref_m)

        cos_a = cos_min_separation(d_omega, d_m)
        sep   = math.degrees(math.acos(cos_a))

        if sep < global_min_sep:
            global_min_sep = sep
            closest_norad  = norad_j
            closest_dOmega = raan_j - ref_raan
            closest_dM     = m_j    - ref_m

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\n  Global minimum angular separation : {global_min_sep:.6f} deg")
    print(f"  Closest pair                      : NORAD {ref_norad} & NORAD {closest_norad}")
    print(f"    dOmega = {closest_dOmega:+.4f} deg")
    print(f"    dM     = {closest_dM:+.4f} deg")

    # Inline verification: also compute the pair's analytical separation here
    cos_check = cos_min_separation(
        math.radians(closest_dOmega),
        math.radians(closest_dM),
    )
    sep_check = math.degrees(math.acos(cos_check))
    print(f"    Verification (recomputed)         : {sep_check:.6f} deg")

    # PASS / FAIL
    passed = global_min_sep >= MIN_SEP_THRESHOLD_DEG
    print()
    print(f"  Threshold : {MIN_SEP_THRESHOLD_DEG:.1f} deg  (Arnas 2D-LFC guarantee)")
    if passed:
        print(f"  Result    : PASS  ({global_min_sep:.4f} deg >= {MIN_SEP_THRESHOLD_DEG:.1f} deg)")
    else:
        print(f"  Result    : FAIL  ({global_min_sep:.4f} deg < {MIN_SEP_THRESHOLD_DEG:.1f} deg)")

    # ------------------------------------------------------------------
    # Additional diagnostics: top-10 closest pairs
    # ------------------------------------------------------------------
    print()
    print("  Top 10 closest pairs (satellite 0 as reference):")
    print(f"    {'NORAD_j':>8}  {'dOmega(deg)':>12}  {'dM(deg)':>10}  {'amin(deg)':>10}")
    print(f"    {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")

    pair_seps = []
    for norad_j, raan_j, m_j in sats[1:]:
        d_omega = math.radians(raan_j - ref_raan)
        d_m     = math.radians(m_j    - ref_m)
        cos_a   = cos_min_separation(d_omega, d_m)
        sep     = math.degrees(math.acos(cos_a))
        pair_seps.append((sep, norad_j, raan_j - ref_raan, m_j - ref_m))

    pair_seps.sort()
    for sep, norad_j, dO, dM in pair_seps[:10]:
        print(f"    {norad_j:>8}  {dO:>12.4f}  {dM:>10.4f}  {sep:>10.6f}")

    # ------------------------------------------------------------------
    # Consistency check: same-plane adjacent satellite
    # ------------------------------------------------------------------
    print()
    print("  Same-plane adjacent satellite check (dOmega=0, dM=40 deg):")
    cos_sp = cos_min_separation(0.0, math.radians(40.0))
    sep_sp = math.degrees(math.acos(cos_sp))
    expected_sp = 40.0
    print(f"    Computed amin  : {sep_sp:.6f} deg")
    print(f"    Expected       : {expected_sp:.6f} deg  (same plane, fixed dM)")
    ok = abs(sep_sp - expected_sp) < 1e-8
    print(f"    Formula check  : {'OK' if ok else 'MISMATCH'}")

    # ------------------------------------------------------------------
    # Consistency check: self-separation (dOmega=0, dM=0) must be 0
    # ------------------------------------------------------------------
    cos_self = cos_min_separation(0.0, 0.0)
    sep_self = math.degrees(math.acos(cos_self))
    ok_self  = abs(sep_self) < 1e-10
    print()
    print(f"  Self-separation check (dOmega=dM=0): {sep_self:.2e} deg"
          f"  ({'OK' if ok_self else 'MISMATCH'})")

    print()
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
