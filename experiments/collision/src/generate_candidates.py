"""
generate_candidates.py
----------------------
Generates a synthetic Shell 3 constellation following Arnas (2021) Table 3.

Constellation parameters (Table 3, i = 30 deg):
    No  = 184   (number of orbital planes)
    Nso = 9     (satellites per plane)
    Nc  = 132   (phasing parameter)
    Total = No * Nso = 1,656 satellites

Orbital slot positions from Arnas (2021) Equation 1:

    [No   0 ] [Omega_ij]         [i-1]
    [Nc  Nso] [M_ij    ]  = 2pi * [j-1]

    where i = 1..No,  j = 1..Nso.

Solving by matrix inverse (det = No*Nso):

    Omega_ij = 2pi * (i-1) / No
    M_ij     = 2pi * ((No*(j-1) - Nc*(i-1)) mod (No*Nso)) / (No*Nso)

Fixed orbital elements (all satellites):
    Altitude    : 550 km  ->  a = 6921 km
    Inclination : 30 deg
    Eccentricity: 0
    Arg of perigee: 0 deg
    Epoch: current UTC

Output:
    data/shell3_synthetic.tle  -- 1,656 two-line element sets
    NORAD IDs: 90001 to 91656

Usage:
    python src/generate_candidates.py
"""

import math
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constellation parameters  (Arnas 2021, Table 3, i = 30 deg)
# ---------------------------------------------------------------------------
NO      = 184       # number of orbital planes
NSO     = 9         # satellites per plane
NC      = 132       # phasing parameter
N_TOTAL = NO * NSO  # 1,656 satellites

# Fixed orbital elements
ALTITUDE_KM     = 550.0
EARTH_RADIUS_KM = 6371.0
A_KM            = EARTH_RADIUS_KM + ALTITUDE_KM   # 6921.0 km
INCLINATION_DEG = 30.0
ECCENTRICITY    = 0.0
ARG_PERIGEE_DEG = 0.0

# Physical constants
MU_KM3_S2 = 398600.4418   # Earth gravitational parameter (km^3/s^2)

# NORAD ID range
NORAD_START = 90001

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "shell3_synthetic.tle"


# ---------------------------------------------------------------------------
# Orbital mechanics helpers
# ---------------------------------------------------------------------------

def mean_motion_rev_per_day(a_km: float) -> float:
    """Keplerian mean motion in revolutions per day for semi-major axis a_km."""
    n_rad_s = math.sqrt(MU_KM3_S2 / a_km ** 3)
    return n_rad_s * 86400.0 / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# TLE formatting helpers
# ---------------------------------------------------------------------------

def tle_checksum(line: str) -> int:
    """
    Compute TLE line checksum over the first 68 characters.

    Rule: sum all digit values; count each '-' as 1; ignore everything else.
    Return sum mod 10.
    """
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == '-':
            total += 1
    return total % 10


def epoch_fields(epoch: datetime) -> tuple[str, str]:
    """
    Convert a UTC datetime to TLE epoch strings.

    Returns
    -------
    y2 : str
        Two-digit year, e.g. '26'.
    doy : str
        Day-of-year with fraction, formatted as DDD.DDDDDDDD (12 chars),
        e.g. '084.50000000'.  Day 1 = 1 January 00:00:00 UTC.
    """
    y2  = f"{epoch.year % 100:02d}"
    soy = datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
    doy = (epoch - soy).total_seconds() / 86400.0 + 1.0
    return y2, f"{doy:012.8f}"


def build_line1(norad_id: int, y2: str, doy: str, elem_set: int) -> str:
    """
    Construct TLE Line 1 (exactly 69 characters including checksum).

    Column layout (1-indexed):
        1       line number  '1'
        2       space
        3-7     NORAD catalog number (5 digits)
        8       classification  'U'
        9       space
        10-17   international designator (8 chars)
        18      space
        19-20   epoch year (2 chars)
        21-32   epoch day-of-year (12 chars: DDD.DDDDDDDD)
        33      space
        34-43   first derivative of mean motion (10 chars)
        44      space
        45-52   second derivative of mean motion (8 chars)
        53      space
        54-61   BSTAR drag term (8 chars)
        62      space
        63      ephemeris type
        64      space
        65-68   element set number (4 chars, right-justified)
        69      checksum digit

    Zero drag (circular, unperturbed orbit): all derivative/BSTAR fields = 0.
    """
    # 34-43: first deriv  " .00000000"  (10 chars: space + dot + 8 digits)
    # 45-52: second deriv " 00000-0"   (8 chars: space + 5 digits + dash + digit)
    # 54-61: BSTAR        " 00000-0"   (8 chars: same format)
    line = (
        "1 "
        + f"{norad_id:05d}"
        + "U "
        + "99001A  "        # international designator placeholder (8 chars)
        + " "               # col 18
        + y2                # cols 19-20
        + doy               # cols 21-32
        + " "               # col 33
        + " .00000000"      # cols 34-43  first deriv
        + " "               # col 44
        + " 00000-0"        # cols 45-52  second deriv
        + " "               # col 53
        + " 00000-0"        # cols 54-61  BSTAR
        + " "               # col 62
        + "0"               # col 63  ephemeris type
        + " "               # col 64
        + f"{elem_set:4d}"  # cols 65-68  element set number
    )
    assert len(line) == 68, f"Line 1 length {len(line)} != 68: {line!r}"
    return line + str(tle_checksum(line))


def build_line2(
    norad_id: int,
    inc_deg: float,
    raan_deg: float,
    ecc: float,
    aop_deg: float,
    ma_deg: float,
    mm_rpd: float,
    rev_num: int = 0,
) -> str:
    """
    Construct TLE Line 2 (exactly 69 characters including checksum).

    Column layout (1-indexed):
        1       line number  '2'
        2       space
        3-7     NORAD catalog number (5 digits)
        8       space
        9-16    inclination (degrees, 8 chars: DDD.DDDD)
        17      space
        18-25   RAAN (degrees, 8 chars: DDD.DDDD)
        26      space
        27-33   eccentricity (7 digits, decimal point assumed)
        34      space
        35-42   argument of perigee (degrees, 8 chars: DDD.DDDD)
        43      space
        44-51   mean anomaly (degrees, 8 chars: DDD.DDDD)
        52      space
        53-63   mean motion (rev/day, 11 chars: DD.DDDDDDDD)
        64-68   revolution number at epoch (5 chars)
        69      checksum digit
    """
    # Eccentricity: 7 digits, no decimal point (e.g. "0000000" for e=0)
    ecc_str = f"{ecc:.7f}"[2:]   # strip leading "0." -> 7-char mantissa

    line = (
        "2 "
        + f"{norad_id:05d}"
        + " "
        + f"{inc_deg:8.4f}"           # cols 9-16
        + " "
        + f"{raan_deg % 360.0:8.4f}"  # cols 18-25
        + " "
        + ecc_str                     # cols 27-33  (7 chars)
        + " "
        + f"{aop_deg % 360.0:8.4f}"   # cols 35-42
        + " "
        + f"{ma_deg % 360.0:8.4f}"    # cols 44-51
        + " "
        + f"{mm_rpd:11.8f}"           # cols 53-63
        + f"{rev_num:05d}"            # cols 64-68
    )
    assert len(line) == 68, f"Line 2 length {len(line)} != 68: {line!r}"
    return line + str(tle_checksum(line))


# ---------------------------------------------------------------------------
# Arnas (2021) Equation 1 — constellation slot solver
# ---------------------------------------------------------------------------

def compute_slots() -> list[tuple[float, float]]:
    """
    Solve for (Omega_ij, M_ij) of all No*Nso orbital slots.

    Arnas (2021) Equation 1:

        [No   0 ] [Omega_ij]         [i-1]
        [Nc  Nso] [M_ij    ]  = 2pi * [j-1]

    Matrix inverse (det = No*Nso, upper-triangular):

        Omega_ij = 2pi * (i-1) / No
        M_ij     = 2pi * ((No*(j-1) - Nc*(i-1)) mod (No*Nso)) / (No*Nso)

    The modulo keeps M_ij in [0, 2pi).  Satellites are enumerated in
    plane-major order: (i=1,j=1), (i=1,j=2), ..., (i=No,j=Nso).

    Returns
    -------
    list of (omega_deg, m_deg), length No*Nso = 1,656.
    """
    two_pi  = 2.0 * math.pi
    n_total = NO * NSO      # 1,656
    slots   = []

    for i in range(1, NO + 1):
        omega_deg = math.degrees(two_pi * (i - 1) / NO)
        for j in range(1, NSO + 1):
            raw   = (NO * (j - 1) - NC * (i - 1)) % n_total
            m_deg = math.degrees(two_pi * raw / n_total)
            slots.append((omega_deg, m_deg))

    return slots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("Arnas (2021) Shell 3 Synthetic Constellation Generator")
    print(f"  No  = {NO:<4d}  (orbital planes)")
    print(f"  Nso = {NSO:<4d}  (satellites per plane)")
    print(f"  Nc  = {NC:<4d}  (phasing parameter)")
    print(f"  Total           = {N_TOTAL}")
    print(f"  Altitude        = {ALTITUDE_KM:.0f} km  (a = {A_KM:.0f} km)")
    print(f"  Inclination     = {INCLINATION_DEG:.1f} deg")
    print(f"  Eccentricity    = {ECCENTRICITY:.1f}")
    print(f"  Arg of perigee  = {ARG_PERIGEE_DEG:.1f} deg")
    print("=" * 62)

    epoch = datetime.now(timezone.utc)
    y2, doy = epoch_fields(epoch)
    mm = mean_motion_rev_per_day(A_KM)

    print(f"\n  Epoch       : {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Mean motion : {mm:.8f} rev/day  (Keplerian, a = {A_KM:.0f} km)")

    # ------------------------------------------------------------------
    # Compute orbital slots via Arnas Eq. 1
    # ------------------------------------------------------------------
    slots = compute_slots()
    assert len(slots) == N_TOTAL

    # ------------------------------------------------------------------
    # Generate TLE pairs
    # ------------------------------------------------------------------
    tle_lines = []
    for idx, (omega_deg, m_deg) in enumerate(slots):
        norad_id = NORAD_START + idx
        elem_set = (idx + 1) % 10000   # TLE field is 4 digits (1-9999, wrap)

        l1 = build_line1(norad_id, y2, doy, elem_set)
        l2 = build_line2(
            norad_id,
            INCLINATION_DEG,
            omega_deg,
            ECCENTRICITY,
            ARG_PERIGEE_DEG,
            m_deg,
            mm,
        )
        tle_lines.append(l1)
        tle_lines.append(l2)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(tle_lines) + "\n")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    omegas = [s[0] for s in slots]
    ms     = [s[1] for s in slots]

    print()
    print("=" * 62)
    print("SANITY CHECKS")
    print("=" * 62)

    print(f"\n  Total satellites  : {len(slots)}  (expected {N_TOTAL})")

    print(f"\n  First 5 satellites (plane i=1, slots j=1..5):")
    print(f"    {'Sat':>4}  {'NORAD':>7}  {'RAAN (deg)':>12}  {'M (deg)':>10}")
    print(f"    {'-'*4}  {'-'*7}  {'-'*12}  {'-'*10}")
    for k in range(5):
        omega, m = slots[k]
        print(f"    {k+1:>4}  {NORAD_START+k:>7}  {omega:>12.4f}  {m:>10.4f}")

    print(f"\n  RAAN range : {min(omegas):.4f} -- {max(omegas):.4f} deg"
          f"  (expected: 0 -- ~{360.0*(NO-1)/NO:.1f})")
    print(f"  M range    : {min(ms):.4f} -- {max(ms):.4f} deg"
          f"  (expected: 0 -- ~360)")

    # Expected RAAN spacing
    raan_spacing = 360.0 / NO
    print(f"\n  Expected RAAN spacing between planes : {raan_spacing:.4f} deg  (360/{NO})")
    # Measured spacing between first two planes
    omega_p1 = slots[0][0]                # plane 1
    omega_p2 = slots[NSO][0]              # plane 2 (after NSO satellites in plane 1)
    print(f"  Measured  RAAN spacing (plane 1->2)  : {omega_p2 - omega_p1:.4f} deg")

    # Expected in-plane M spacing
    m_spacing = 360.0 / NSO
    print(f"\n  Expected in-plane M spacing  : {m_spacing:.4f} deg  (360/{NSO})")
    m_p1 = [slots[j][1] for j in range(NSO)]
    m_diffs = [abs(m_p1[j+1] - m_p1[j]) for j in range(NSO - 1)]
    print(f"  Measured M spacing (plane 1) : {m_diffs[0]:.4f} deg")

    # ------------------------------------------------------------------
    # SGP4 parse verification (sample 10 satellites)
    # ------------------------------------------------------------------
    print()
    try:
        from sgp4.api import Satrec, jday

        jd_check, fr_check = jday(
            epoch.year, epoch.month, epoch.day,
            epoch.hour, epoch.minute,
            epoch.second + epoch.microsecond / 1e6
        )

        n_check = 10
        ok, err = 0, 0
        for k in range(0, n_check * 2, 2):
            sat = Satrec.twoline2rv(tle_lines[k], tle_lines[k + 1])
            e, r, v = sat.sgp4(jd_check, fr_check)
            if e == 0:
                ok += 1
            else:
                err += 1

        # Spot-check altitude of first satellite at epoch
        sat0 = Satrec.twoline2rv(tle_lines[0], tle_lines[1])
        _, r0, _ = sat0.sgp4(jd_check, fr_check)
        alt0 = math.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2) - EARTH_RADIUS_KM

        print(f"  SGP4 parse check (first {n_check}) : {ok} OK / {err} errors")
        print(f"  Altitude of sat 90001 at epoch     : {alt0:.1f} km  (expected ~{ALTITUDE_KM:.0f} km)")

    except ImportError:
        print("  SGP4 parse check : sgp4 library not available, skipping")

    print(f"\n  Saved {len(slots)} TLE objects to: {OUTPUT_PATH}")
    print("=" * 62)
    print("Done.")


if __name__ == "__main__":
    main()
