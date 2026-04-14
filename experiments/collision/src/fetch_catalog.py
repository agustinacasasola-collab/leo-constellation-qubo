"""
fetch_catalog.py
----------------
Downloads the full conjunction-threat catalog from Space-Track.org.

Filter criteria (Owens-Fahrner et al. 2025):
    APOAPSIS  < 2000 km   — keeps all objects whose highest point is below
                             2000 km, capturing the complete LEO threat environment
                             while excluding MEO/GEO objects
    ECCENTRICITY < 1.0    — bound orbits only (hyperbolic objects excluded)

Expected yield: ~22,000+ objects (active satellites, rocket bodies, debris).

Credentials are loaded from the .env file in the project root:
    SPACETRACK_USER=your_email@example.com
    SPACETRACK_PASS=your_password

Usage:
    python src/fetch_catalog.py
"""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
BASE_URL = "https://www.space-track.org"

# Catalog filter thresholds (Owens-Fahrner 2025)
APOAPSIS_MAX_KM = 2000.0   # apogee altitude upper bound (km)
ECCENTRICITY_MAX = 1.0     # eccentricity upper bound (bound orbits only)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "leo_catalog.tle"


def load_credentials() -> tuple[str, str]:
    """
    Load Space-Track credentials from the .env file.

    Returns
    -------
    tuple of (username, password)

    Raises
    ------
    EnvironmentError
        If credentials are not found.
    """
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    user = os.getenv("SPACETRACK_USER")
    password = os.getenv("SPACETRACK_PASS")

    if not user or not password:
        raise EnvironmentError(
            "Space-Track credentials not found.\n"
            "Create a .env file in the project root with:\n"
            "  SPACETRACK_USER=your_email@example.com\n"
            "  SPACETRACK_PASS=your_password\n"
            "Register for free at https://www.space-track.org/auth/createAccount"
        )

    return user, password


def build_query_url() -> str:
    """
    Build the Space-Track GP query URL for the conjunction-threat catalog.

    Filters (Owens-Fahrner 2025):
    - APOAPSIS < 2000 km     — apogee altitude below 2000 km (LEO + upper LEO)
    - ECCENTRICITY < 1.0     — bound orbits only
    - EPOCH within last 30 days — recent TLEs only
    - No limit (downloads all matching objects; expected ~22,000+)
    - Format: TLE

    Space-Track URL encoding:
        %3C = '<'   (APOAPSIS less-than filter)

    Returns
    -------
    str
        Full query URL.
    """
    epoch_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%d"
    )

    url = (
        f"{BASE_URL}/basicspacedata/query/class/gp"
        f"/APOAPSIS/%3C{APOAPSIS_MAX_KM:.0f}"       # APOAPSIS < 2000 km
        f"/ECCENTRICITY/%3C{ECCENTRICITY_MAX:.1f}"  # ECCENTRICITY < 1.0
        f"/EPOCH/%3E{epoch_cutoff}"
        f"/orderby/NORAD_CAT_ID asc"
        f"/format/tle"
    )
    return url


def fetch_catalog() -> str:
    """
    Authenticate with Space-Track and download the full LEO catalog.

    Uses a persistent requests.Session so the login cookie is reused
    for the data query.

    Returns
    -------
    str
        Raw TLE text content.

    Raises
    ------
    EnvironmentError
        If credentials are missing.
    requests.HTTPError
        If login or data request fails.
    PermissionError
        If login credentials are rejected.
    """
    user, password = load_credentials()

    with requests.Session() as session:
        # Authenticate
        print("Authenticating with Space-Track.org...")
        resp = session.post(
            LOGIN_URL,
            data={"identity": user, "password": password},
            timeout=30,
        )
        resp.raise_for_status()

        if "Failed" in resp.text or "invalid" in resp.text.lower():
            raise PermissionError(
                "Space-Track login failed. Check SPACETRACK_USER and "
                "SPACETRACK_PASS in the .env file."
            )
        print("  Login successful.")

        # Download full LEO catalog (no limit — may take a minute)
        query_url = build_query_url()
        print(f"  Querying: {query_url}")
        print("  Downloading full LEO catalog (no limit — this may take a moment)...")
        resp = session.get(query_url, timeout=300)
        resp.raise_for_status()

        return resp.text


def save_catalog(tle_text: str, output_path: Path) -> tuple[int, dict, str, str]:
    """
    Save TLE text to file, returning count, object-type breakdown, and epoch range.

    Returns
    -------
    num_objects : int
    type_counts : dict  {type_char: count}  where type_char is col 7 of Line 1:
        'U' = unclassified (most objects), 'C' = classified, 'S' = secret
        (Space-Track also embeds object-type in name lines when 3LE format is
        used; in 2LE format the classification character gives a rough proxy.)
    epoch_oldest : str   ISO-formatted UTC of oldest epoch in catalog
    epoch_newest : str   ISO-formatted UTC of newest epoch in catalog
    """
    import math

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [line for line in tle_text.strip().splitlines() if line.strip()]
    # Keep only valid TLE line pairs
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        l1, l2 = lines[i], lines[i + 1]
        if l1.startswith('1 ') and l2.startswith('2 '):
            pairs.append((l1, l2))

    num_objects = len(pairs)

    # Write 2-line pairs
    with open(output_path, "w") as f:
        for l1, l2 in pairs:
            f.write(l1 + "\n")
            f.write(l2 + "\n")

    # Parse epochs for range and classification counts
    epochs_jd = []
    type_counts: dict[str, int] = {}

    for l1, _ in pairs:
        # Classification character (col 7, 0-indexed)
        cls = l1[7] if len(l1) > 7 else 'U'
        type_counts[cls] = type_counts.get(cls, 0) + 1

        # Epoch: year (cols 18-19) + day-of-year fraction (cols 20-31)
        try:
            y2  = int(l1[18:20])
            doy = float(l1[20:32])
            yr  = (2000 + y2) if y2 < 57 else (1900 + y2)
            # Approximate JD for range comparison
            epochs_jd.append(yr + (doy - 1) / 365.25)
        except (ValueError, IndexError):
            pass

    epoch_oldest = epoch_newest = "N/A"
    if epochs_jd:
        def _epoch_str(approx_yr: float) -> str:
            yr  = int(approx_yr)
            doy = (approx_yr - yr) * 365.25 + 1
            try:
                d = datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
                return d.strftime("%Y-%m-%d")
            except Exception:
                return str(approx_yr)
        epoch_oldest = _epoch_str(min(epochs_jd))
        epoch_newest = _epoch_str(max(epochs_jd))

    return num_objects, type_counts, epoch_oldest, epoch_newest


def main() -> None:
    print("=" * 65)
    print("Space-Track Catalog Downloader  (Owens-Fahrner 2025, Sec 4.1)")
    print(f"  APOAPSIS     < {APOAPSIS_MAX_KM:.0f} km  (LEO only)")
    print(f"  ECCENTRICITY < {ECCENTRICITY_MAX:.1f}   (bound orbits)")
    print(f"  EPOCH        > now-30 days  (recent TLEs only)")
    print(f"  Object types : ALL  (debris, payloads, rocket bodies, fragments)")
    print("=" * 65)

    tle_text = fetch_catalog()
    num_saved, type_counts, epoch_oldest, epoch_newest = save_catalog(
        tle_text, OUTPUT_PATH
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  Total objects downloaded : {num_saved:,}")
    print(f"  File size                : {size_mb:.2f} MB")
    print(f"  Epoch range              : {epoch_oldest}  ..  {epoch_newest}")
    print(f"  Classification breakdown :")
    for cls, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        label = {'U': 'Unclassified', 'C': 'Classified', 'S': 'Secret'}.get(cls, cls)
        print(f"    {label:15s} : {cnt:,}")
    print(f"  Saved to                 : {OUTPUT_PATH}")
    if num_saved < 20000:
        print(f"\n  NOTE: expected ~20,000-26,000 objects; got {num_saved:,}.")
        print("  Check EPOCH cutoff or Space-Track catalog freshness.")
    print("\nDone.")


if __name__ == "__main__":
    main()
