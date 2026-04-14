"""
generate_walker53.py  (Step 1 -- Walker-53 Experiment)
-------------------------------------------------------
Generate a Walker Delta T=648 / P=72 / F=1 constellation at 53 deg
inclination, 550 km altitude, as a TLE set for use with SGP4.

Walker Delta formulas
---------------------
    Planes:       P = 72
    Sats/plane:   S = T/P = 9
    RAAN_i      = i * (360 / P)                       [deg]
    M_{i,j}     = j * (360 / S) + i * F * (360 / T)  [deg]

NORAD IDs: 95001 -- 95648  (plane i, satellite j: 95001 + i*9 + j)
Epoch:     Inherited from data/shell3_synthetic.tle  (line 1, field 19-32)
Mean motion: 15.07819960 rev/day  (550 km circular, same as shell3_synthetic)

Output:
    experiments/walker_53/data/walker53.tle
"""

import os
import sys
from math import radians, cos, sin, sqrt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data')
CATALOG_TLE = os.path.join(ROOT, 'data', 'shell3_synthetic.tle')
OUTPUT_TLE  = os.path.join(DATA_DIR, 'walker53.tle')

# ---------------------------------------------------------------------------
# Walker Delta parameters
# ---------------------------------------------------------------------------
T   = 648       # total satellites
P   = 72        # orbital planes
S   = T // P    # 9 satellites per plane
F   = 1         # phasing parameter

INC_DEG    = 53.0          # inclination
ALT_KM     = 550.0         # altitude (circular orbit)
MEAN_MOT   = 15.07819960   # rev/day at 550 km (matches shell3_synthetic.tle)
ECCENTR    = 0.0000000     # circular orbit
ARG_PERIG  = 0.0           # deg (undefined for circular; set to 0)

NORAD_BASE = 95001
INTL_DESIG = '25001A  '    # 8-char international designator (field 10-17)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tle_checksum(line: str) -> int:
    """Compute TLE line checksum (sum of digits, '-' counts as 1, mod 10)."""
    total = 0
    for ch in line:
        if ch.isdigit():
            total += int(ch)
        elif ch == '-':
            total += 1
    return total % 10


def read_epoch_from_tle(tle_path: str) -> str:
    """Return the epoch string (14 chars, YYDDD.DDDDDDDD) from the first TLE."""
    with open(tle_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('1 '):
                # TLE line 1: columns 18-31 (0-indexed) = epoch field
                return line[18:32].strip()
    raise ValueError(f"No TLE line 1 found in {tle_path}")


def fmt_float(value: float, width: int, decimals: int) -> str:
    """Right-justified fixed-point string with no leading sign for positive."""
    fmt = f'{{:{width}.{decimals}f}}'
    return fmt.format(value)


def make_tle_line1(norad: int, epoch: str, elem_num: int = 1) -> str:
    """
    Construct TLE Line 1.

    Format (1-indexed columns):
      1       : '1'
      3-7     : NORAD (5 digits, zero-padded)
      8       : 'U'
      10-17   : international designator (8 chars)
      19-32   : epoch (14 chars)
      34-43   : first deriv mean motion '.00000000'
      45-52   : second deriv '00000-0'
      54-61   : BSTAR '00000-0'
      63      : '0'
      65-68   : element set number
      69      : checksum
    """
    norad_str = f'{norad:05d}'
    # epoch field is exactly 14 chars; pad with spaces if needed
    epoch_field = f'{epoch:<14s}'
    elem_str = f'{elem_num:4d}'

    core = (f'1 {norad_str}U {INTL_DESIG} {epoch_field}  '
            f'.00000000  00000-0  00000-0 0 {elem_str}')
    cs = tle_checksum(core)
    return core + str(cs)


def make_tle_line2(norad: int, inc: float, raan: float,
                   ecc: float, argp: float, ma: float, mm: float,
                   rev_num: int = 0) -> str:
    """
    Construct TLE Line 2.

    Format (1-indexed columns):
      1       : '2'
      3-7     : NORAD (5 digits)
      9-16    : inclination (8 chars)
      18-25   : RAAN (8 chars)
      27-33   : eccentricity (7 digits, no decimal point)
      35-42   : argument of perigee (8 chars)
      44-51   : mean anomaly (8 chars)
      53-63   : mean motion (11 chars)
      64-68   : revolution number (5 chars)
      69      : checksum
    """
    norad_str = f'{norad:05d}'
    inc_str   = fmt_float(inc,  8, 4)
    raan_str  = fmt_float(raan, 8, 4)
    # eccentricity: 7 digits, no decimal point (implied at start)
    ecc_int   = round(ecc * 1e7)
    ecc_str   = f'{ecc_int:07d}'
    argp_str  = fmt_float(argp, 8, 4)
    ma_str    = fmt_float(ma,   8, 4)
    mm_str    = f'{mm:11.8f}'
    rev_str   = f'{rev_num:05d}'

    core = (f'2 {norad_str} {inc_str} {raan_str} {ecc_str} '
            f'{argp_str} {ma_str} {mm_str}{rev_str}')
    cs = tle_checksum(core)
    return core + str(cs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print('=' * 65)
    print('STEP 1 -- Generate Walker-53 TLEs')
    print('=' * 65)
    print(f'\n  Walker Delta  T={T} / P={P} / F={F}')
    print(f'  Inclination   {INC_DEG} deg')
    print(f'  Altitude      {ALT_KM} km  (circular)')
    print(f'  NORAD range   {NORAD_BASE} -- {NORAD_BASE + T - 1}')

    # -- Read epoch from shell3_synthetic.tle ----------------------------------
    if not os.path.exists(CATALOG_TLE):
        print(f'\n  ERROR: {CATALOG_TLE} not found.')
        sys.exit(1)
    epoch = read_epoch_from_tle(CATALOG_TLE)
    print(f'  Epoch         {epoch}  (from {os.path.basename(CATALOG_TLE)})')

    # -- Generate TLEs ---------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    n_written = 0

    with open(OUTPUT_TLE, 'w') as out:
        for i in range(P):
            raan = i * (360.0 / P)           # RAAN for plane i (deg)
            for j in range(S):
                ma = (j * (360.0 / S) + i * F * (360.0 / T)) % 360.0
                norad = NORAD_BASE + i * S + j
                elem_num = n_written + 1

                l1 = make_tle_line1(norad, epoch, elem_num)
                l2 = make_tle_line2(norad, INC_DEG, raan, ECCENTR,
                                    ARG_PERIG, ma, MEAN_MOT, rev_num=0)
                out.write(l1 + '\n')
                out.write(l2 + '\n')
                n_written += 1

    print(f'\n  Satellites written : {n_written}')
    print(f'  Output             : {OUTPUT_TLE}')

    # -- Verify first and last TLE -------------------------------------------
    with open(OUTPUT_TLE) as f:
        lines = f.read().splitlines()

    print(f'\n  First TLE:')
    print(f'    {lines[0]}')
    print(f'    {lines[1]}')
    print(f'\n  Last TLE:')
    print(f'    {lines[-2]}')
    print(f'    {lines[-1]}')

    # -- Sanity: RAAN range --------------------------------------------------
    raan_max = (P - 1) * (360.0 / P)
    ma_max   = ((S - 1) * (360.0 / S) + (P - 1) * F * (360.0 / T)) % 360.0
    print(f'\n  RAAN range     : 0.0 -- {raan_max:.4f} deg  ({P} planes x {360/P:.2f} deg)')
    print(f'  MA last sat    : {ma_max:.4f} deg')
    print('\nDone.')


if __name__ == '__main__':
    main()
