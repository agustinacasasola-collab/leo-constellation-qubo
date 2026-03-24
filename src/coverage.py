"""
coverage.py
-----------
Coverage and observability constraint functions for LEO satellite analysis.

Implements geometric access constraints used to determine whether a satellite
can observe a catalog object at a given time step. In a full production
implementation these functions would be called at every propagation time step.

Reference:
    Owens-Fahrner, N., Wysack, J., Kim, J. (2025). Graph-Based Optimization
    for High-Density LEO Constellation Design. AMOS Conference.
"""

import math
import numpy as np
from typing import List


def check_solar_exclusion(
    viewer_pos: np.ndarray,
    target_pos: np.ndarray,
    sun_pos: np.ndarray,
    min_angle_deg: float = 30.0,
) -> bool:
    """
    Check whether a target is outside the solar exclusion zone.

    The solar exclusion constraint prevents optical sensors from observing
    targets that are too close in angle to the Sun, which would saturate
    detectors or degrade signal quality.

    Access is granted (returns True) if the angle between the
    viewer-to-Sun vector and the viewer-to-target vector is greater than
    ``min_angle_deg``.

    Parameters
    ----------
    viewer_pos : np.ndarray, shape (3,)
        Position of the observing satellite in an inertial frame (km).
    target_pos : np.ndarray, shape (3,)
        Position of the target object (km).
    sun_pos : np.ndarray, shape (3,)
        Position of the Sun (km). Typically from a solar ephemeris.
    min_angle_deg : float, optional
        Minimum allowed Sun-viewer-target angle in degrees. Default 30°,
        consistent with Owens-Fahrner et al. (2025).

    Returns
    -------
    bool
        True if target is observable (outside exclusion zone),
        False if inside the solar exclusion cone.
    """
    vec_to_sun = sun_pos - viewer_pos
    vec_to_target = target_pos - viewer_pos

    norm_sun = np.linalg.norm(vec_to_sun)
    norm_target = np.linalg.norm(vec_to_target)

    if norm_sun < 1e-10 or norm_target < 1e-10:
        return False

    # Cosine of the angle between the two unit vectors.
    cos_angle = np.dot(vec_to_sun, vec_to_target) / (norm_sun * norm_target)
    # Clamp to [-1, 1] to guard against floating-point rounding.
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))

    return angle_deg > min_angle_deg


def check_earth_limb(
    viewer_pos: np.ndarray,
    target_pos: np.ndarray,
    earth_radius_km: float = 6371.0,
    min_grazing_angle_deg: float = 5.0,
) -> bool:
    """
    Check whether a target is above the Earth limb as seen from the viewer.

    The Earth-limb constraint rejects observations where the line of sight
    passes too close to the Earth's surface (large atmospheric path length
    degrades optical tracking performance).

    Access is granted (returns True) if the grazing angle — the elevation
    of the target above the local horizon defined by Earth's limb — exceeds
    ``min_grazing_angle_deg``.

    Parameters
    ----------
    viewer_pos : np.ndarray, shape (3,)
        Position of the observing satellite in an inertial frame (km).
    target_pos : np.ndarray, shape (3,)
        Position of the target object (km).
    earth_radius_km : float, optional
        Mean Earth radius (km). Default 6371 km.
    min_grazing_angle_deg : float, optional
        Minimum grazing angle above limb in degrees. Default 5°,
        consistent with Owens-Fahrner et al. (2025).

    Returns
    -------
    bool
        True if the target is above the limb constraint,
        False if the line of sight clips the Earth.
    """
    los = target_pos - viewer_pos
    los_norm = np.linalg.norm(los)
    if los_norm < 1e-10:
        return False

    los_unit = los / los_norm

    # Closest approach distance from the origin (Earth's centre) to the
    # line-of-sight ray: d = |viewer_pos × los_unit|.
    cross = np.cross(viewer_pos, los_unit)
    min_distance_km = np.linalg.norm(cross)

    # Convert minimum distance to an elevation angle above the limb.
    # If min_distance_km >= earth_radius_km the LOS never enters Earth.
    if min_distance_km >= earth_radius_km:
        return True

    # Grazing angle: how far above the surface is the closest LOS point?
    grazing_angle_deg = math.degrees(
        math.asin(min_distance_km / earth_radius_km)
    ) - 90.0 + 90.0  # elevation above limb

    # Simpler direct computation: angle at Earth centre between limb and LOS.
    grazing_angle_deg = math.degrees(math.asin(min_distance_km / earth_radius_km))
    limb_angle_deg = 90.0 - grazing_angle_deg

    return limb_angle_deg > min_grazing_angle_deg


def compute_coverage_fraction(satellite_data: dict) -> float:
    """
    Return the coverage fraction for a candidate satellite.

    Coverage fraction is defined as the proportion of (time step, catalog
    object) pairs during the simulation window for which the candidate
    satellite has valid access to the catalog object — i.e., all five
    access constraints are simultaneously satisfied:

    1. **Line of sight** — no Earth occultation between satellite and object.
    2. **Solar exclusion** — Sun-viewer-target angle > 30° (check_solar_exclusion).
    3. **Earth limb exclusion** — grazing angle > 5° (check_earth_limb).
    4. **Target sunlit** — target object is in sunlight (not in Earth's shadow).
    5. **Minimum SNR** — signal-to-noise ratio > 6.0, a function of target
       size, range, relative velocity, and solar phase angle.

    .. note::
        This implementation returns the pre-computed ``'coverage'`` value
        directly from ``satellite_data``. A full production implementation
        would propagate ephemerides for both the candidate satellite and all
        filtered catalog objects using an orbital propagator (SGP4, Orekit,
        or GMAT), evaluate all five constraints at each time step, and
        compute the fraction of passing evaluations. The helper functions
        ``check_solar_exclusion`` and ``check_earth_limb`` support that
        future integration.

    Parameters
    ----------
    satellite_data : dict
        Dictionary containing at minimum the key ``'coverage'`` (float),
        the pre-computed coverage fraction for this satellite.

    Returns
    -------
    float
        Coverage fraction in [0, 1]. Higher values indicate the satellite
        can observe a greater proportion of the LEO catalog over the
        simulation period.
    """
    return float(satellite_data['coverage'])
