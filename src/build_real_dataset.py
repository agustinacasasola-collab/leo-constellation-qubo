"""
build_real_dataset.py
----------------------
Merges real collision probabilities (satellite_pc.csv) with TLE-derived
coverage fractions into a single data/real_satellites.csv ready for
the QUBO graph construction pipeline.

Coverage model
--------------
For a circular orbit at inclination i, the fraction of Earth's surface
whose latitude falls within ±i degrees is sin(i).  This is the physical
ground coverage fraction for a satellite that can observe objects anywhere
along its swath.

    coverage = sin(inclination_rad)

Values range from sin(32°) ≈ 0.53 (NORAD 01835, equatorial-ish) to
sin(98°) ≈ 0.99 (SSO satellites).

Usage:
    python src/build_real_dataset.py
"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.collision import load_tle_pairs, parse_tle_line2


DATA_DIR = Path(__file__).parent.parent / "data"
PC_CSV = DATA_DIR / "satellite_pc.csv"
TLE_PATH = DATA_DIR / "shell_550km.tle"
OUTPUT_CSV = DATA_DIR / "real_satellites.csv"


def main() -> None:
    # Load computed Pc values
    pc_df = pd.read_csv(PC_CSV)
    pc_df["norad_id"] = pc_df["norad_id"].astype(str).str.zfill(5)

    # Extract inclination and altitude from TLEs
    tle_rows = []
    for l1, l2 in load_tle_pairs(TLE_PATH):
        band = parse_tle_line2(l1, l2)
        inclination_deg = float(l2[8:16])
        mean_alt_km = (band.apogee_km + band.perigee_km) / 2.0
        tle_rows.append({
            "norad_id": band.norad_id,
            "inclination_deg": inclination_deg,
            "mean_alt_km": mean_alt_km,
        })
    tle_df = pd.DataFrame(tle_rows)

    # Merge on norad_id — keep only satellites that have both a TLE and a Pc
    merged = pc_df.merge(tle_df, on="norad_id", how="inner")
    print(f"  Satellites with both Pc and TLE : {len(merged)}")

    # Coverage fraction: sin(inclination_rad)
    # Physical interpretation: proportion of Earth's surface within ±i latitude.
    merged["coverage"] = merged["inclination_deg"].apply(
        lambda inc: math.sin(math.radians(inc))
    )

    # Build output DataFrame matching graph_builder.py expectations
    out = pd.DataFrame({
        "satellite_id": merged["norad_id"],
        "pc": merged["aggregate_pc"],
        "coverage": merged["coverage"],
        "altitude_km": merged["mean_alt_km"],
        "inclination_deg": merged["inclination_deg"],
        "num_conjunctions": merged["num_conjunctions"],
        "min_tca_km": merged["min_tca_km"],
    })

    out = out.sort_values("satellite_id").reset_index(drop=True)

    print("\n  Real satellite dataset:")
    print(f"  {'NORAD':>8}  {'Pc':>12}  {'Coverage':>9}  "
          f"{'Alt(km)':>8}  {'Inc(deg)':>9}  {'Close<100km':>12}  {'MinTCA(km)':>11}")
    for _, row in out.iterrows():
        print(f"  {row['satellite_id']:>8}  {row['pc']:>12.3e}  "
              f"{row['coverage']:>9.4f}  {row['altitude_km']:>8.1f}  "
              f"{row['inclination_deg']:>9.3f}  {row['num_conjunctions']:>12.0f}  "
              f"{row['min_tca_km']:>11.2f}")

    print(f"\n  Pc range    : {out['pc'].min():.3e} – {out['pc'].max():.3e}")
    print(f"  Coverage range: {out['coverage'].min():.4f} – {out['coverage'].max():.4f}")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
