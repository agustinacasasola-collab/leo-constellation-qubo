"""
generate_shell_a.py
-------------------
Generates Shell A Walker constellation (single-shell, single-satellite-per-plane).

Shell parameters:
    Altitude : 550 km
    Inc      : 53.0 deg
    N_PLANES : 200
    RAAN_i   = i * (360 / 200)  for i = 0..199
    e = 0, omega = 0, M = 0 (all satellites at ascending node at epoch)

NORAD IDs: 93001 to 93200

Outputs:
    data/shell_a_candidates.tle
    data/shell_a_candidates.csv   columns: norad_id, alt_km, inc_deg, raan_deg

Usage:
    python src/generate_shell_a.py
"""

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constellation parameters
# ---------------------------------------------------------------------------
ALT_KM    = 550.0
INC_DEG   = 53.0
N_PLANES  = 200
NORAD_START = 93001

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_TLE  = DATA_DIR / "shell_a_candidates.tle"
OUTPUT_CSV  = DATA_DIR / "shell_a_candidates.csv"

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
    ecc_str = "0000000"   # e = 0
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
        + f"{0.0:8.4f}"              # omega = 0
        + " "
        + f"{ma_deg % 360.0:8.4f}"   # mean anomaly
        + " "
        + f"{mm_rpd:11.8f}"
        + f"{0:05d}"                  # rev number 0
    )
    assert len(line) == 68, f"Line 2 length {len(line)} != 68"
    return line + str(tle_checksum(line))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("Shell A Candidate Generator")
    print(f"  Alt      : {ALT_KM} km")
    print(f"  Inc      : {INC_DEG} deg")
    print(f"  N_PLANES : {N_PLANES}")
    print(f"  NORAD    : {NORAD_START} - {NORAD_START + N_PLANES - 1}")
    print("=" * 55)

    epoch = datetime.now(timezone.utc)
    y2, doy = epoch_fields(epoch)
    print(f"\n  Epoch : {epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    a_km   = EARTH_RADIUS_KM + ALT_KM
    mm_rpd = mean_motion_rev_per_day(a_km)

    tle_lines = []
    csv_rows  = []

    for i in range(N_PLANES):
        norad_id = NORAD_START + i
        raan_deg = i * (360.0 / N_PLANES)

        l1 = build_line1(norad_id, y2, doy, elem_set=i + 1)
        ma_deg = i * (360.0 / N_PLANES)   # distribute phase across orbit
        l2 = build_line2(norad_id, INC_DEG, raan_deg, mm_rpd, ma_deg=ma_deg)
        tle_lines.extend([l1, l2])

        csv_rows.append({
            "norad_id": norad_id,
            "alt_km":   ALT_KM,
            "inc_deg":  INC_DEG,
            "raan_deg": raan_deg,
            "ma_deg":   ma_deg,
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TLE.write_text("\n".join(tle_lines) + "\n")
    print(f"\n  Wrote {len(tle_lines) // 2} TLEs to : {OUTPUT_TLE.name}")

    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Wrote {len(df)} rows to  : {OUTPUT_CSV.name}")

    # Gate
    assert len(df) == N_PLANES, f"GATE FAILED: expected {N_PLANES} rows, got {len(df)}"
    print(f"\n  GATE PASS: {len(df)} == {N_PLANES}  OK")
    print("Done.")


if __name__ == "__main__":
    main()
