# SPDX-License-Identifier: Apache-2.0
"""Stage 6: game-engine and GIS exports from capture geometry.

The corridor is not just a viewer leash: it is the racing line. Where the
camera went is, by construction, where a car can go -- so the same polyline
that bounds the reconstruction also gives a sim its drivable ribbon, its
collision surface, and (once you drive it in-game) its AI spline.

Emitted from `world/corridor.json`:

    road.obj        swept ribbon along each pass, ready for Blender/ksEditor
    centerline.obj  the bare polylines
    centerline.csv  x,y,z + lat,lon,alt per point
    centerline.geojson  WGS84 LineStrings for GIS / trailworks

Nothing here is sim-specific on purpose. Assetto Corsa wants FBX through
ksEditor with `1ROAD_`-prefixed meshes, BeamNG wants DAE, RBR wants its own
chain -- all of them start from an OBJ, and all of them generate AI lines by
driving the track rather than by importing a spline.
"""

import json
import math
from pathlib import Path

from .geo import enu_to_ll


def _axis(v, y_up: bool):
    """ENU (east, north, up) -> Y-up right-handed (x, y, z) if requested.

    Most DCC and engine importers (OBJ/FBX/glTF) default to Y-up, while our
    world frame is Z-up. Mapping (e, n, u) -> (e, u, -n) keeps it right-handed
    so faces do not wind inside-out.
    """
    e, n, u = v
    return (e, u, -n) if y_up else (e, n, u)


def _ribbon(points, width_m: float):
    """Two edge rails offset +-width/2 from a polyline, in the XY plane.

    Offsets use the average of adjacent segment directions, so corners stay
    continuous instead of splitting into disjoint quads.
    """
    n = len(points)
    dirs = []
    for i in range(n - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        length = math.hypot(dx, dy) or 1.0
        dirs.append((dx / length, dy / length))
    left, right = [], []
    half = width_m / 2
    for i, p in enumerate(points):
        d0 = dirs[max(i - 1, 0)]
        d1 = dirs[min(i, len(dirs) - 1)]
        dx, dy = d0[0] + d1[0], d0[1] + d1[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length          # left normal
        left.append((p[0] + nx * half, p[1] + ny * half, p[2]))
        right.append((p[0] - nx * half, p[1] - ny * half, p[2]))
    return left, right


def export(world_dir: Path, out: Path | None = None, width_m: float = 6.0,
           y_up: bool = True, drop_m: float = 2.4) -> Path:
    """Write road/centerline geometry from a merged world's corridor.

    `drop_m` lowers the ribbon from camera height to the road surface: the
    corridor records where the *lens* was, which on a roof rig is ~2.4 m above
    the tarmac. Measure yours once and pass it.
    """
    world_dir = Path(world_dir)
    out = Path(out or world_dir / "export")
    out.mkdir(parents=True, exist_ok=True)
    corridor = json.loads((world_dir / "corridor.json").read_text())
    origin = (corridor["origin"]["lat"], corridor["origin"]["lon"], 0.0)

    road_v, road_f, line_v, line_e = [], [], [], []
    csv_rows = ["pass,index,x_enu,y_enu,z_enu,lat,lon,alt"]
    features = []

    for pi, pas in enumerate(corridor.get("passes", [])):
        pts = [(p[0], p[1], p[2] - drop_m) for p in pas["points"]]
        if len(pts) < 2:
            continue
        left, right = _ribbon(pts, width_m)

        base = len(road_v) + 1                      # OBJ is 1-indexed
        for l, r in zip(left, right):
            road_v.append(_axis(l, y_up))
            road_v.append(_axis(r, y_up))
        for i in range(len(pts) - 1):
            a, b = base + i * 2, base + i * 2 + 1
            c, d = a + 2, b + 2
            road_f.append((a, c, d))                # two tris per quad
            road_f.append((a, d, b))

        lbase = len(line_v) + 1
        coords = []
        for i, p in enumerate(pts):
            line_v.append(_axis(p, y_up))
            lat, lon, alt = enu_to_ll(p[0], p[1], p[2], origin)
            csv_rows.append(f"{pi},{i},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},"
                            f"{lat:.8f},{lon:.8f},{alt:.2f}")
            coords.append([round(lon, 8), round(lat, 8), round(alt, 2)])
        line_e += [(lbase + i, lbase + i + 1) for i in range(len(pts) - 1)]
        features.append({"type": "Feature",
                         "properties": {"pass": pi, "chunk": pas.get("chunk"),
                                        "video": pas.get("video")},
                         "geometry": {"type": "LineString", "coordinates": coords}})

    up = "Y-up (e, u, -n)" if y_up else "Z-up ENU"
    with open(out / "road.obj", "w") as fh:
        fh.write(f"# gaussworks road ribbon: width {width_m} m, {up}\n")
        fh.write(f"o road\n")
        for v in road_v:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in road_f:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")
    with open(out / "centerline.obj", "w") as fh:
        fh.write(f"# gaussworks centerline, {up}\n")
        for v in line_v:
            fh.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for a, b in line_e:
            fh.write(f"l {a} {b}\n")
    (out / "centerline.csv").write_text("\n".join(csv_rows) + "\n")
    (out / "centerline.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=1))

    length = sum(math.dist(pas["points"][i], pas["points"][i + 1])
                 for pas in corridor.get("passes", [])
                 for i in range(len(pas["points"]) - 1))
    print(f"[export] {len(features)} passes, {length / 1000:.2f} km centerline, "
          f"{len(road_v)} verts / {len(road_f)} tris ({up}) -> {out}")
    return out
