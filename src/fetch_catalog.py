"""
fetch_catalog.py
----------------
Downloads the full LEO catalog from Space-Track.org.

Queries all objects with MEAN_MOTION > 11.25 rev/day, corresponding to
altitudes below ~1000 km (Low Earth Orbit). This includes active satellites,
rocket bodies, and debris fragments — the complete threat environment for
conjunction analysis.

MEAN_MOTION > 11.25 rev/day derivation (circular orbit approximation):
    h = 1000 km  →  a = 7371 km  →  n ≈ 11.25 rev/day

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

# Minimum mean motion for LEO (objects below ~1000 km)
MEAN_MOTION_MIN = 11.25

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
    Build the Space-Track GP query URL for the full LEO catalog.

    Filters:
    - MEAN_MOTION > 11.25 rev/day (altitude below ~1000 km)
    - EPOCH within the last 30 days (recent TLEs only)
    - Ordered by NORAD_CAT_ID ascending
    - No limit (downloads all available objects)
    - Format: TLE

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
        f"/MEAN_MOTION/%3E{MEAN_MOTION_MIN}"   # MEAN_MOTION > 11.25
        f"/EPOCH/%3E{epoch_cutoff}"             # EPOCH > cutoff date
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


def save_catalog(tle_text: str, output_path: Path) -> int:
    """
    Save TLE text to file and return the number of objects saved.

    Parameters
    ----------
    tle_text : str
        Raw TLE content from Space-Track.
    output_path : Path
        Destination file path.

    Returns
    -------
    int
        Number of TLE objects (line pairs) saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [line for line in tle_text.strip().splitlines() if line.strip()]
    num_objects = len(lines) // 2

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return num_objects


def main() -> None:
    print("=" * 60)
    print("Space-Track Full LEO Catalog Downloader")
    print(f"MEAN_MOTION filter: > {MEAN_MOTION_MIN} rev/day (alt < ~1000 km)")
    print("=" * 60)

    tle_text = fetch_catalog()
    num_saved = save_catalog(tle_text, OUTPUT_PATH)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  Total objects downloaded : {num_saved:,}")
    print(f"  File size                : {size_mb:.2f} MB")
    print(f"  Saved to                 : {OUTPUT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
