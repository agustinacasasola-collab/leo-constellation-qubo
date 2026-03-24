"""
fetch_tles.py
-------------
Downloads TLE data from Space-Track.org for LEO objects near 550 km altitude.

Queries the GP class filtering by MEAN_MOTION between 15.04 and 15.12 rev/day,
which corresponds to the ~540-560 km altitude band centered on the Starlink
Shell 3 at 550 km — the most congested region in the current LEO environment.

Derivation (vis-viva / Kepler's third law, circular orbit):
    n = sqrt(GM / a^3)   [rad/s]
    a = R_earth + h
    GM = 398600.4418 km^3/s^2,  R_earth = 6371 km

    h = 540 km  →  a = 6911 km  →  n ≈ 15.11 rev/day
    h = 550 km  →  a = 6921 km  →  n ≈ 15.08 rev/day
    h = 560 km  →  a = 6931 km  →  n ≈ 15.05 rev/day

Credentials are loaded from a .env file (never hardcoded):
    SPACETRACK_USER=your_email@example.com
    SPACETRACK_PASS=your_password

Usage:
    python src/fetch_tles.py
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

# MEAN_MOTION range for ~540-560 km altitude (rev/day)
# Derived from vis-viva: n = sqrt(GM / a^3), converted to rev/day
# h=560 km → 15.05 rev/day,  h=540 km → 15.11 rev/day
MEAN_MOTION_MIN = 15.04  # slight margin below 560 km
MEAN_MOTION_MAX = 15.12  # slight margin above 540 km

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "shell_550km.tle"


def load_credentials() -> tuple[str, str]:
    """
    Load Space-Track credentials from the .env file.

    The .env file must be in the project root and contain:
        SPACETRACK_USER=your_email
        SPACETRACK_PASS=your_password

    Returns
    -------
    tuple of (username, password)

    Raises
    ------
    EnvironmentError
        If credentials are not found in the environment.
    """
    # Look for .env in the project root (two levels up from src/)
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


def build_query_url(limit: int = 20) -> str:
    """
    Build the Space-Track GP query URL.

    Filters:
    - MEAN_MOTION between MEAN_MOTION_MIN and MEAN_MOTION_MAX (550 km shell)
    - EPOCH within the last 30 days (recent TLEs only)
    - Ordered by NORAD_CAT_ID ascending
    - Format: TLE (two-line element set, plain text)

    Parameters
    ----------
    limit : int
        Maximum number of TLE objects to return.

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
        f"/MEAN_MOTION/{MEAN_MOTION_MIN}--{MEAN_MOTION_MAX}"
        f"/EPOCH/%3E{epoch_cutoff}"   # EPOCH > cutoff date
        f"/orderby/NORAD_CAT_ID asc"
        f"/limit/{limit}"
        f"/format/tle"
    )
    return url


def fetch_tles(limit: int = 20) -> str:
    """
    Authenticate with Space-Track and download TLE data.

    Uses a persistent requests.Session so the login cookie is reused
    for the data query — Space-Track requires authentication via POST
    before any data endpoint is accessible.

    Parameters
    ----------
    limit : int
        Maximum number of TLE objects to fetch.

    Returns
    -------
    str
        Raw TLE text content (pairs of lines, one object per 2 lines).

    Raises
    ------
    EnvironmentError
        If credentials are missing from .env.
    requests.HTTPError
        If login or data request fails.
    """
    user, password = load_credentials()

    with requests.Session() as session:
        # Step 1: authenticate — POST credentials to login endpoint
        print("Authenticating with Space-Track.org...")
        login_payload = {"identity": user, "password": password}
        resp = session.post(LOGIN_URL, data=login_payload, timeout=30)
        resp.raise_for_status()

        if "Failed" in resp.text or "invalid" in resp.text.lower():
            raise PermissionError(
                "Space-Track login failed. Check your SPACETRACK_USER and "
                "SPACETRACK_PASS in the .env file."
            )
        print("  Login successful.")

        # Step 2: query TLE data using the authenticated session
        query_url = build_query_url(limit=limit)
        print(f"  Querying: {query_url}")
        resp = session.get(query_url, timeout=60)
        resp.raise_for_status()

        return resp.text


def save_tles(tle_text: str, output_path: Path) -> int:
    """
    Save TLE text to file and return the number of objects saved.

    Each TLE object occupies exactly 2 lines. Empty lines are stripped
    before counting.

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
    print("Space-Track TLE Fetcher — Shell 3 (~550 km)")
    print(f"MEAN_MOTION filter: {MEAN_MOTION_MIN} -- {MEAN_MOTION_MAX} rev/day")
    print("=" * 60)

    tle_text = fetch_tles(limit=30)
    num_saved = save_tles(tle_text, OUTPUT_PATH)

    print(f"\n  Saved {num_saved} TLE objects to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
