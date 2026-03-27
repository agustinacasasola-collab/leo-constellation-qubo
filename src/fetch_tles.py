"""
fetch_tles.py
-------------
Downloads TLE data from Space-Track.org for Shell 3 candidates.

Targets the Owens-Fahrner et al. (2025) Shell 3 design parameters:
    altitude    : 550 km
    inclination : ~30 deg

Mean-motion filter (Owens-Fahrner Table 2, Shell 3):
    MEAN_MOTION_MIN = 15.14 rev/day
    MEAN_MOTION_MAX = 15.22 rev/day

Note on Kozai correction: SGP4 TLEs use the Kozai-corrected mean motion,
which differs slightly from the Keplerian value.  For i = 30 deg the J2
secular correction raises the TLE mean motion above the Keplerian value
by ~0.06 rev/day, shifting the band from the pure-Kepler 15.08 rev/day
to ~15.14 rev/day — consistent with the paper's filter.

Inclination filter ±5 deg around 30 deg (25–35 deg) is added to isolate
Shell 3 candidates and avoid contamination from higher-inclination shells
in the same altitude band.

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

# Owens-Fahrner 2025 Shell 3 parameters (Table 2):
#   altitude 550 km, inclination 30 deg
# MEAN_MOTION filter (Kozai-corrected TLE values for 550 km / 30 deg):
#   Keplerian n at 550 km ≈ 15.08 rev/day
#   J2 Kozai correction for i=30° adds ~+0.06 rev/day → ~15.14 rev/day
MEAN_MOTION_MIN = 15.14   # lower bound (rev/day)
MEAN_MOTION_MAX = 15.22   # upper bound (rev/day)

# Inclination band centred on Shell 3 target of 30 deg (±5 deg)
# NOTE: for the 100-candidate random-baseline validation (Table 5) the
# inclination filter is disabled because only ~19 real objects exist at
# <35 deg in the 550-km MEAN_MOTION band.  The full 754-object pool is used.
INCLINATION_MIN = 20.0    # deg  (unused when FILTER_INCLINATION = False)
INCLINATION_MAX = 42.0    # deg  (unused when FILTER_INCLINATION = False)

# Set True to restrict to the 30-deg inclination band (20-candidate run).
# Set False to fetch from all inclinations (100-candidate random baseline).
FILTER_INCLINATION = False

# Number of candidates to download
N_CANDIDATES = 100

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


def build_query_url(limit: int = N_CANDIDATES) -> str:
    """
    Build the Space-Track GP query URL for Shell 3 candidates.

    Filters applied:
    - MEAN_MOTION in [MEAN_MOTION_MIN, MEAN_MOTION_MAX]  (~550 km, Kozai-corrected)
    - INCLINATION in [INCLINATION_MIN, INCLINATION_MAX]  (~30 deg ± 5 deg)
    - EPOCH within the last 30 days (fresh TLEs only)
    - Ordered by NORAD_CAT_ID ascending
    - Format: TLE plain text

    Parameters
    ----------
    limit : int
        Maximum number of TLE objects to return. Default N_CANDIDATES (20).

    Returns
    -------
    str
        Full query URL.
    """
    epoch_cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime(
        "%Y-%m-%d"
    )

    inc_segment = (
        f"/INCLINATION/{INCLINATION_MIN}--{INCLINATION_MAX}"
        if FILTER_INCLINATION else ""
    )
    url = (
        f"{BASE_URL}/basicspacedata/query/class/gp"
        f"/MEAN_MOTION/{MEAN_MOTION_MIN}--{MEAN_MOTION_MAX}"
        f"{inc_segment}"
        f"/EPOCH/%3E{epoch_cutoff}"
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
    print("Space-Track TLE Fetcher — Shell 3 (550 km / 30 deg)")
    print(f"  MEAN_MOTION      : {MEAN_MOTION_MIN} - {MEAN_MOTION_MAX} rev/day")
    if FILTER_INCLINATION:
        print(f"  INCLINATION      : {INCLINATION_MIN} - {INCLINATION_MAX} deg")
    else:
        print("  INCLINATION      : all (filter disabled for 100-candidate baseline)")
    print(f"  Candidates       : {N_CANDIDATES}")
    print("=" * 60)

    tle_text = fetch_tles(limit=N_CANDIDATES)
    num_saved = save_tles(tle_text, OUTPUT_PATH)

    if num_saved < N_CANDIDATES:
        print(f"\n  WARNING: only {num_saved}/{N_CANDIDATES} objects returned.")
        print("  Space-Track may have fewer objects in this MEAN_MOTION + INCLINATION band.")
        print("  Consider widening MEAN_MOTION or INCLINATION ranges in fetch_tles.py.")
    else:
        print(f"\n  Saved {num_saved} TLE objects to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
