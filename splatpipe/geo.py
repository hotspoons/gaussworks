# SPDX-License-Identifier: Apache-2.0
"""Small geodesy helpers: local tangent-plane distances from lat/lon."""

import math

R_EARTH = 6378137.0


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
