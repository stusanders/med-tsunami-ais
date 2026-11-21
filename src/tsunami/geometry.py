"""Shared geometry conventions (see docs/interfaces.md section 2).

Angles in radians. Headings, lane directions and strikes are compass bearings:
clockwise from true north. Vectors are (east, north).
"""
from .backend import array_module


def heading_unit(theta):
    """Unit vector along compass bearing theta: (east, north) = (sin, cos)."""
    m = array_module(theta)
    return m.sin(theta), m.cos(theta)


def heading_normal(theta):
    """Starboard unit normal for heading theta: (east, north) = (cos, -sin).

    A current u.n > 0 pushes the ship to starboard, i.e. SOG*sin(COG-HDG) > 0.
    """
    m = array_module(theta)
    return m.cos(theta), -m.sin(theta)


def project_normal(u, v, theta):
    """Heading-normal (starboard-positive) component of current (u east, v north)."""
    m = array_module(theta)
    return u * m.cos(theta) - v * m.sin(theta)


def fold_axis(theta):
    """Fold a direction to an undirected axis in [0, pi)."""
    m = array_module(theta)
    return m.mod(theta, m.pi)


def haversine_m(lon1_deg, lat1_deg, lon2_deg, lat2_deg, radius_m):
    """Great-circle distance in metres; inputs in degrees."""
    m = array_module(lon1_deg)
    p1, p2 = m.deg2rad(lat1_deg), m.deg2rad(lat2_deg)
    dphi = p2 - p1
    dlam = m.deg2rad(lon2_deg) - m.deg2rad(lon1_deg)
    a = m.sin(dphi / 2) ** 2 + m.cos(p1) * m.cos(p2) * m.sin(dlam / 2) ** 2
    return 2 * radius_m * m.arcsin(m.sqrt(a))


def cell_index(lon_deg, lat_deg, cell_m, ref_lat_deg, lon0_deg, lat0_deg, radius_m):
    """Integer (cj, ci) of the ~cell_m scoring/traffic cell containing each point.

    Equirectangular projection about ref_lat_deg; origin at (lon0, lat0) = domain SW corner.
    Used by scoring (T15) and traffic Poisson areas (T04) so both see the same cells.
    """
    m = array_module(lon_deg)
    y = radius_m * m.deg2rad(lat_deg - lat0_deg)
    x = radius_m * m.cos(m.deg2rad(ref_lat_deg)) * m.deg2rad(lon_deg - lon0_deg)
    return m.floor(y / cell_m).astype(int), m.floor(x / cell_m).astype(int)
