"""
generate_shell3r.py  --  STEP 1
---------------------------------
Generates 130 Shell-3-Reduced (shell3r) Walker-LFC candidates.

Parameters (Arnas 2D-LFC, Nso=1):
    N_PLANES  = 130
    ALT_KM    = 550     (Arnas Shell 3 altitude)
    INC_DEG   = 53.0    (Arnas Shell 3 inclination)
    RAAN_i    = i * (360 / 130)  for i = 0..129
    M_i       = 0.0 for all     (ascending node at epoch)
    ecc       = 0.0, omega = 0.0

NORAD IDs: 94001 to 94130 (distinct from all previous ranges)

Epoch: read from data/shell3_synthetic.tle to ensure timestep
alignment with data/propagated_catalog.csv.
Falls back to current UTC if that file is absent.

Outputs:
    data/shell3r_candidates.tle
    data/shell3r_candidates.csv
      Columns: norad_id, alt_km, inc_deg, raan_deg, mean_anomaly_deg

Usage:
    python src/generate_shell3r.py
"""

import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Parameters — single source of truth for the shell3r pipeline
# ---------------------------------------------------------------------------
N_PLANES   = 130
ALT_KM     = 550.0
INC_DEG    = 53.0
NORAD_BASE = 94001

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR         = Path(__file__).parent.parent / "data"
SHELL3_TLE_PATH  = DATA_DIR / "shell3_synthetic.tle"
OUTPUT_TLE_PATH  = DATA_DIR / "shell3r_candidates.tle"
OUTPUT_CSV_PATH  = DATA_DIR / "shell3r_candidates.csv"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
MU_KM3_S2       = 398600.4418


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_motion_rev_per_day(a_km: float) -> float:
    n_rad_s = math.sqrt(MU_KM3_S2 / a_km ** 3)
    return n_rad_s * 86400.0 / (2.0 * math.pi)


def tle_checksum(line: str) -> int:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == '-':
            total += 1
    return total % 10


def epoch_fields(epoch: datetime) -> tuple[str, str]:
    y2  = f"{epoch.year % 100:02d}"
    soy = datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
    doy = (epoch - soy).total_seconds() / 86400.0 + 1.0
    return y2, f"{doy:012.8f}"


def build_line1(norad_id: int, y2: str, doy: str, elem_set: int) -> str:
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
    mm_rpd: float,
    ma_deg: float = 0.0,
) -> str:
    ecc_str = "0000000"  # e = 0
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
        + f"{0.0:8.4f}"            # omega = 0
        + " "
        + f"{ma_deg % 360.0:8.4f}" # mean anomaly
        + " "
        + f"{mm_rpd:11.8f}"
        + f"{0:05d}"               # rev number
    )
    assert len(line) == 68, f"Line 2 length {len(line)} != 68"
    return line + str(tle_checksum(line))


def read_shell3_epoch() -> datetime | None:
    """Read epoch from the first TLE in shell3_synthetic.tle for alignment."""
    if not SHELL3_TLE_PATH.exists():
        return None
    lines = [ln.strip() for ln in SHELL3_TLE_PATH.read_text().splitlines() if ln.strip()]
    if len(lines) < 1:
        return None
    l1 = lines[0]
    if not l1.startswith('1 '):
        return None
    try:
        y2_ref  = int(l1[18:20])
        doy_ref = float(l1[20:32])
        year    = (2000 + y2_ref) if y2_ref < 57 else (1900 + y2_ref)
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy_ref - 1.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raan_spacing = 360.0 / N_PLANES

    print("=" * 58)
    print("Shell 3 Reduced (shell3r) Candidate Generator")
    print(f"  Shell 3 reduced: {N_PLANES} candidates")
    print(f"  Altitude: {ALT_KM:.0f} km, Inclination: {INC_DEG:.1f} deg")
    print(f"  RAAN spacing: {raan_spacing:.2f} deg")
    print(f"  NORAD range: {NORAD_BASE}-{NORAD_BASE + N_PLANES - 1}")
    print("=" * 58)

    # --- Epoch ---------------------------------------------------------------
    epoch = read_shell3_epoch()
    if epoch is not None:
        print(f"\n  Epoch source : shell3_synthetic.tle")
        print(f"  Epoch        : {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  (Aligned with propagated_catalog.csv)")
    else:
        epoch = datetime.now(timezone.utc)
        print(f"\n  WARNING: shell3_synthetic.tle not found.")
        print(f"  Using current UTC epoch: {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  This may cause misalignment with propagated_catalog.csv.")

    y2, doy = epoch_fields(epoch)
    a_km    = EARTH_RADIUS_KM + ALT_KM
    mm_rpd  = mean_motion_rev_per_day(a_km)

    # --- Generate TLEs and CSV rows ------------------------------------------
    tle_lines = []
    csv_rows  = []

    for i in range(N_PLANES):
        norad_id = NORAD_BASE + i
        raan_deg = i * raan_spacing

        l1 = build_line1(norad_id, y2, doy, elem_set=i + 1)
        l2 = build_line2(norad_id, INC_DEG, raan_deg, mm_rpd, ma_deg=0.0)
        tle_lines.extend([l1, l2])

        csv_rows.append({
            "norad_id":         norad_id,
            "alt_km":           ALT_KM,
            "inc_deg":          INC_DEG,
            "raan_deg":         raan_deg,
            "mean_anomaly_deg": 0.0,
        })

    # --- Write outputs -------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TLE_PATH.write_text("\n".join(tle_lines) + "\n")
    print(f"\n  Wrote {len(tle_lines) // 2} TLEs to : {OUTPUT_TLE_PATH.name}")

    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"  Wrote {len(df)} rows to  : {OUTPUT_CSV_PATH.name}")

    # --- Gate ----------------------------------------------------------------
    assert len(df) == N_PLANES, (
        f"GATE FAILED: expected {N_PLANES} rows, got {len(df)}"
    )
    print(f"\n  GATE PASS: {len(df)} == {N_PLANES}  OK")
    print("Done.")


if __name__ == "__main__":
    main()
