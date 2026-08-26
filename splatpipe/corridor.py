# SPDX-License-Identifier: Apache-2.0
"""Capture corridors: where the camera actually was, per chunk.

A corridor is the polyline(s) the camera travelled plus a radius and height
band. It is the geometric statement of "this is the part of the world we
observed", and everything downstream wants it:

- viewers leash the camera to it, so you cannot fly into unobserved void
- the game takes it as the drivable centreline (and collision seed)
- gaussians outside the envelope are floaters and can be pruned
- gaps in it are the roads the capture missed

Coordinates are local ENU metres against a stated lat/lon origin, so corridors
from different chunks compose into one world.
"""

from .geo import ll_to_enu


def _runs(frames, origin, gap_m, gap_s=30.0):
    """Split a chunk's frames into contiguous camera passes.

    A chunk gathers frames from every pass through its cell, so the points are
    not one path: they are several, and consecutive frames in the file can be
    minutes and hundreds of metres apart. Cut wherever the jump is too large
    to be one continuous drive, and keep passes from different videos apart.

    Distance alone is not enough. A `--near` fence is left and re-entered on
    the same road, so the last frame of one visit and the first of the next
    can be metres apart in space and minutes apart in time: the street run's
    three visits in one chapter collapsed into a single "pass" that way. A
    jump in video time (`t`) larger than gap_s is a new pass too.
    """
    lat0, lon0 = origin
    out, cur, prev = [], [], None
    for f in frames:
        x, y, _z = ll_to_enu(f["lat"], f["lon"], float(f.get("alt") or 0.0),
                             (lat0, lon0, 0.0))
        pt = (x, y, _z, f)
        if prev is not None:
            far = ((x - prev[0]) ** 2 + (y - prev[1]) ** 2) ** 0.5 > gap_m
            other_video = f.get("video") != prev[3].get("video")
            t0, t1 = prev[3].get("t"), f.get("t")
            late = t0 is not None and t1 is not None and abs(t1 - t0) > gap_s
            if far or other_video or late:
                out.append(cur)
                cur = []
        cur.append(pt)
        prev = pt
    if cur:
        out.append(cur)
    return out


def _decimate(points, min_spacing_m):
    """Thin a pass to one point per min_spacing_m (endpoints always kept)."""
    if len(points) < 3 or min_spacing_m <= 0:
        return points
    kept = [points[0]]
    for p in points[1:-1]:
        q = kept[-1]
        if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 >= min_spacing_m:
            kept.append(p)
    kept.append(points[-1])
    return kept


def build(frames, origin, radius_m=25.0, height_margin_m=8.0,
          min_spacing_m=5.0, gap_m=30.0, gap_s=30.0) -> dict:
    """Corridor for one chunk's frames. `origin` is the world (lat, lon)."""
    passes = []
    for run in _runs(frames, origin, gap_m, gap_s):
        pts = _decimate(run, min_spacing_m)
        if len(pts) < 2:
            continue
        passes.append({
            "video": pts[0][3].get("video"),
            "seq_range": [pts[0][3]["seq"], pts[-1][3]["seq"]],
            "points": [[round(p[0], 3), round(p[1], 3), round(p[2], 3)] for p in pts],
        })
    zs = [p[2] for pas in passes for p in
          [(pt[0], pt[1], pt[2]) for pt in pas["points"]]] or [0.0]
    return {
        "frame": "enu",
        "origin": {"lat": origin[0], "lon": origin[1]},
        "radius_m": radius_m,
        "height_band_m": [min(zs) - height_margin_m, max(zs) + height_margin_m],
        "passes": passes,
    }
