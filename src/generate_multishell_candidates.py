"""
generate_multishell_candidates.py
----------------------------------
Generates a 3-shell Walker constellation inspired by Starlink Phase 1.
One satellite per plane (Nso=1).  Coverage is NOT included here.

Shells:
    A: 550 km, 53.0°
    B: 560 km, 70.0°
    C: 560 km, 97.6°

Change N_PLANES_PER_SHELL below to scale the constellation:
    6  → 18 total candidates  (Level 1, default)
    12 → 36 total candidates  (Level 2)

RAAN spacing: RAAN_i = i * (360 / N_PLANES_PER_SHELL),  i = 0..N_PLANES_PER_SHELL-1
All other orbital elements zero (e=0, omega=0, M=0).

Synthetic NORAD IDs start at 92001 (distinct from Shell 3 range 90001-91656).

Epoch note:
    If data/shell3_synthetic.tle exists, its epoch is reused for the new TLEs
    so that propagated_catalog.csv (generated from that epoch) aligns correctly
    in compute_pc_multishell.py.  Otherwise the current UTC time is used.

Outputs:
    data/multishell_candidates.tle
    data/multishell_candidates.csv   columns: norad_id, shell_label, alt_km,
                                              inc_deg, raan_deg

Usage:
    python src/generate_multishell_candidates.py
"""

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# ============================================================================
# ► SCALE HERE: change this single value to go from Level 1 (N=18) to
#               Level 2 (N=36) or beyond.
# ============================================================================
N_PLANES_PER_SHELL = 200  # satellites per shell  ->  total = N_PLANES_PER_SHELL

# ---------------------------------------------------------------------------
# Single-shell configuration (53 deg / 550 km -- Starlink-like broadband shell)
# Filter tolerances for downstream pipeline consistency checks:
#   inc: +/-1 deg,  alt: +/-20 km
# ---------------------------------------------------------------------------
SINGLE_SHELL_INC_DEG = 53.0
SINGLE_SHELL_ALT_KM  = 550
INC_TOL_DEG          = 1.0
ALT_TOL_KM           = 20.0

SHELLS = [
    {"alt_km": SINGLE_SHELL_ALT_KM, "inc_deg": SINGLE_SHELL_INC_DEG, "label": "A"},
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR              = Path(__file__).parent.parent / "data"
SHELL3_TLE_PATH       = DATA_DIR / "shell3_synthetic.tle"
OUTPUT_TLE_PATH       = DATA_DIR / "multishell_candidates.tle"
OUTPUT_CSV_PATH       = DATA_DIR / "multishell_candidates.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
MU_KM3_S2       = 398600.4418
NORAD_START     = 92001   # distinct from Shell 3 range 90001-91656
NSO             = 1       # satellites per plane (POC: one per plane)


# ---------------------------------------------------------------------------
# Helpers (replicated from generate_candidates.py for standalone operation)
# ---------------------------------------------------------------------------

def mean_motion_rev_per_day(a_km: float) -> float:
    """Keplerian mean motion in rev/day for semi-major axis a_km."""
    n_rad_s = math.sqrt(MU_KM3_S2 / a_km ** 3)
    return n_rad_s * 86400.0 / (2.0 * math.pi)


def tle_checksum(line: str) -> int:
    """TLE line checksum: sum digits + count '-' as 1, mod 10."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == '-':
            total += 1
    return total % 10


def epoch_fields(epoch: datetime) -> tuple[str, str]:
    """Return (y2, doy_str) for the TLE epoch field."""
    y2  = f"{epoch.year % 100:02d}"
    soy = datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
    doy = (epoch - soy).total_seconds() / 86400.0 + 1.0
    return y2, f"{doy:012.8f}"


def build_line1(norad_id: int, y2: str, doy: str, elem_set: int) -> str:
    """Construct TLE Line 1 (69 chars including checksum)."""
    line = (
        "1 "
        + f"{norad_id:05d}"
        + "U "
        + "99001A  "
        + " "
        + y2
        + doy
        + " "
        + " .00000000"
        + " "
        + " 00000-0"
        + " "
        + " 00000-0"
        + " "
        + "0"
        + " "
        + f"{elem_set:4d}"
    )
    assert len(line) == 68, f"Line 1 length {len(line)} != 68"
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
    """Construct TLE Line 2 (69 chars including checksum)."""
    ecc_str = f"{ecc:.7f}"[2:]   # strip "0." → 7-char mantissa
    line = (
        "2 "
        + f"{norad_id:05d}"
        + " "
        + f"{inc_deg:8.4f}"
        + " "
        + f"{raan_deg % 360.0:8.4f}"
        + " "
        + ecc_str
        + " "
        + f"{aop_deg % 360.0:8.4f}"
        + " "
        + f"{ma_deg % 360.0:8.4f}"
        + " "
        + f"{mm_rpd:11.8f}"
        + f"{rev_num:05d}"
    )
    assert len(line) == 68, f"Line 2 length {len(line)} != 68"
    return line + str(tle_checksum(line))


def read_shell3_epoch() -> datetime | None:
    """
    Read the epoch from the first TLE in shell3_synthetic.tle (if it exists).
    Returns None if the file is missing.
    """
    if not SHELL3_TLE_PATH.exists():
        return None
    lines = [ln.strip() for ln in SHELL3_TLE_PATH.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    l1 = lines[0]
    try:
        y2_ref  = int(l1[18:20])
        doy_ref = float(l1[20:32])
        year    = (2000 + y2_ref) if y2_ref < 57 else (1900 + y2_ref)
        return (datetime(year, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=doy_ref - 1.0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_total = len(SHELLS) * N_PLANES_PER_SHELL

    print("=" * 62)
    print("Multi-Shell Candidate Generator")
    print(f"  N_PLANES_PER_SHELL = {N_PLANES_PER_SHELL}")
    print(f"  Shells             = {len(SHELLS)}  ({', '.join(s['label'] for s in SHELLS)})")
    print(f"  Total candidates   = {n_total}")
    print("=" * 62)

    # --- Choose epoch -------------------------------------------------------
    shell3_epoch = read_shell3_epoch()
    if shell3_epoch is not None:
        epoch = shell3_epoch
        print(f"\n  Epoch source : shell3_synthetic.tle  "
              f"(aligns with propagated_catalog.csv)")
    else:
        epoch = datetime.now(timezone.utc)
        print(f"\n  Epoch source : current UTC time")
        print(f"  WARNING: shell3_synthetic.tle not found - epoch may not align "
              f"with propagated_catalog.csv")

    print(f"  Epoch        : {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    y2, doy = epoch_fields(epoch)

    # --- Generate TLEs and CSV rows -----------------------------------------
    tle_lines  = []
    csv_rows   = []
    norad_id   = NORAD_START

    for shell in SHELLS:
        alt_km  = shell["alt_km"]
        inc_deg = shell["inc_deg"]
        label   = shell["label"]
        a_km    = EARTH_RADIUS_KM + alt_km
        mm      = mean_motion_rev_per_day(a_km)

        for i in range(N_PLANES_PER_SHELL):
            raan_deg = i * (360.0 / N_PLANES_PER_SHELL)

            l1 = build_line1(norad_id, y2, doy, elem_set=norad_id - NORAD_START + 1)
            l2 = build_line2(
                norad_id,
                inc_deg  = inc_deg,
                raan_deg = raan_deg,
                ecc      = 0.0,
                aop_deg  = 0.0,
                ma_deg   = 0.0,
                mm_rpd   = mm,
            )
            tle_lines.extend([l1, l2])

            csv_rows.append({
                "norad_id":   norad_id,
                "shell_label": label,
                "alt_km":     alt_km,
                "inc_deg":    inc_deg,
                "raan_deg":   raan_deg,
            })
            norad_id += 1

    # --- Write outputs -------------------------------------------------------
    OUTPUT_TLE_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_TLE_PATH.write_text("\n".join(tle_lines) + "\n")
    print(f"\n  Wrote {len(tle_lines)//2} TLEs to: {OUTPUT_TLE_PATH.name}")

    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"  Wrote {len(df)} rows to:   {OUTPUT_CSV_PATH.name}")

    # --- Summary ------------------------------------------------------------
    print()
    for shell in SHELLS:
        label   = shell["label"]
        alt_km  = shell["alt_km"]
        inc_deg = shell["inc_deg"]
        n = N_PLANES_PER_SHELL
        print(f"  Shell {label} ({alt_km} km, {inc_deg}deg): {n} candidates")
    print(f"  Total: {n_total} candidates")

    # --- Gate ---------------------------------------------------------------
    expected_rows = len(SHELLS) * N_PLANES_PER_SHELL
    assert len(df) == expected_rows, (
        f"GATE FAILED: expected {expected_rows} rows, got {len(df)}"
    )
    print(f"\n  GATE PASS: {len(df)} == {len(SHELLS)} x {N_PLANES_PER_SHELL}  OK")
    print("Done.")


if __name__ == "__main__":
    main()
