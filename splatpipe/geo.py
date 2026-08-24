# SPDX-License-Identifier: Apache-2.0
"""Small geodesy helpers: local tangent-plane distances from lat/lon."""

import math

R_EARTH = 6378137.0          # WGS84 semi-major axis
_F = 1 / 298.257223563       # WGS84 flattening
_E2 = _F * (2 - _F)          # first eccentricity squared


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> tuple[float, float, float]:
    """Geodetic -> Earth-Centred Earth-Fixed metres (WGS84)."""
    la, lo = math.radians(lat), math.radians(lon)
    sin_la, cos_la = math.sin(la), math.cos(la)
    n = R_EARTH / math.sqrt(1 - _E2 * sin_la * sin_la)
    return ((n + alt) * cos_la * math.cos(lo),
            (n + alt) * cos_la * math.sin(lo),
            (n * (1 - _E2) + alt) * sin_la)


def ll_to_enu(lat: float, lon: float, alt: float,
              origin: tuple[float, float, float]) -> tuple[float, float, float]:
    """Geodetic -> local ENU metres about one project origin (lat, lon, alt).

    This is THE world frame: every chunk's poses (and therefore its gaussians)
    are aligned to it, so merging chunks is concatenation rather than a pile of
    per-chunk transforms. Exact for the whole capture area, unlike the
    small-angle projection in ll_to_xy.
    """
    lat0, lon0, alt0 = origin
    x, y, z = lla_to_ecef(lat, lon, alt)
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    la, lo = math.radians(lat0), math.radians(lon0)
    sin_la, cos_la, sin_lo, cos_lo = (math.sin(la), math.cos(la),
                                      math.sin(lo), math.cos(lo))
    return (-sin_lo * dx + cos_lo * dy,
            -sin_la * cos_lo * dx - sin_la * sin_lo * dy + cos_la * dz,
            cos_la * cos_lo * dx + cos_la * sin_lo * dy + sin_la * dz)


def ll_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular local projection (meters), fine at road-network scale."""
    x = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R_EARTH
    return x, y


def track_distances(lats: list[float], lons: list[float]) -> list[float]:
    """Cumulative along-track distance in meters."""
    if not lats:
        return []
    lat0, lon0 = lats[0], lons[0]
    xys = [ll_to_xy(la, lo, lat0, lon0) for la, lo in zip(lats, lons)]
    dist = [0.0]
    for (x0, y0), (x1, y1) in zip(xys, xys[1:]):
        dist.append(dist[-1] + math.hypot(x1 - x0, y1 - y0))
    return dist


def ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF metres -> geodetic (lat, lon, alt). Ferrari's closed-form solution."""
    a, e2 = R_EARTH, _E2
    b2 = a * a * (1 - e2)
    r = math.hypot(x, y)
    if r < 1e-9:                                  # on the polar axis
        return (90.0 if z >= 0 else -90.0), 0.0, abs(z) - math.sqrt(b2)
    ep2 = (a * a - b2) / b2
    f = 54 * b2 * z * z
    g = r * r + (1 - e2) * z * z - e2 * (a * a - b2)
    c = e2 * e2 * f * r * r / (g ** 3)
    ss = (1 + c + math.sqrt(max(c * c + 2 * c, 0.0))) ** (1 / 3)
    pp = f / (3 * (ss + 1 / ss + 1) ** 2 * g * g)
    q = math.sqrt(1 + 2 * e2 * e2 * pp)
    r0 = (-(pp * e2 * r) / (1 + q)
          + math.sqrt(max(0.5 * a * a * (1 + 1 / q)
                          - pp * (1 - e2) * z * z / (q * (1 + q)) - 0.5 * pp * r * r, 0.0)))
    u = math.hypot(r - e2 * r0, z)
    v = math.hypot(r - e2 * r0, z * math.sqrt(1 - e2))
    z0 = b2 * z / (a * v)
    return (math.degrees(math.atan((z + ep2 * z0) / r)),
            math.degrees(math.atan2(y, x)),
            u * (1 - b2 / (a * v)))


def enu_to_ll(e: float, n: float, u: float,
              origin: tuple[float, float, float]) -> tuple[float, float, float]:
    """Inverse of ll_to_enu: local ENU metres -> (lat, lon, alt).

    Lets any world-frame geometry (corridors, road meshes, tiles) be written
    back out as real coordinates for GIS, or for trailworks to consume.
    """
    lat0, lon0, alt0 = origin
    la, lo = math.radians(lat0), math.radians(lon0)
    sin_la, cos_la, sin_lo, cos_lo = (math.sin(la), math.cos(la),
                                      math.sin(lo), math.cos(lo))
    dx = -sin_lo * e - sin_la * cos_lo * n + cos_la * cos_lo * u
    dy = cos_lo * e - sin_la * sin_lo * n + cos_la * sin_lo * u
    dz = cos_la * n + sin_la * u
    x0, y0, z0 = lla_to_ecef(lat0, lon0, alt0)
    return ecef_to_lla(x0 + dx, y0 + dy, z0 + dz)
