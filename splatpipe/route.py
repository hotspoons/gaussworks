# SPDX-License-Identifier: Apache-2.0
"""Stage 6b: pick a driveable STAGE out of a captured corridor.

A capture is deliberately redundant -- every road driven both ways, every
intersection crossed repeatedly -- because redundancy is what makes the
reconstruction good. A stage is the opposite: one direction, one sequence of
roads, a start and a finish. Point-to-point is a first-class layout in
Assetto Corsa (Trento-Bondone is 17.3 km of it), so this turns the corridor's
pile of overlapping passes into a single ordered centreline.

Two steps, both deliberately simple and inspectable:

1. DEDUPE -- drop passes that merely retrace a road already chosen (the
   opposite-direction twin), so the stage runs one way down each road.
2. CHAIN -- greedily join the remaining passes end-to-end, starting from the
   most isolated endpoint (a dead end reads as a natural stage start), joining
   only where endpoints are within `join_m` of each other.

Write the result with `export --route`, or hand-edit route.json: it is just an
ordered list of pass indices, and flipping the sign reverses a pass. Handmade
beats clever here -- you know which way down Bell Branch is more fun.
"""

import json
import math
from pathlib import Path


def _polyline(pas, flip=False):
    pts = [tuple(p) for p in pas["points"]]
    return pts[::-1] if flip else pts


def _length(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _covered_by(pts, chosen, tol_m):
    """Fraction of pts lying within tol_m of any already-chosen polyline."""
    if not chosen:
        return 0.0
    near = 0
    for p in pts:
        for other in chosen:
            if any(math.dist(p, q) <= tol_m for q in other):
                near += 1
                break
    return near / max(len(pts), 1)


def build(corridor: dict, join_m: float = 60.0, dedupe_tol_m: float = 20.0,
          dedupe_frac: float = 0.6, min_length_m: float = 50.0) -> dict:
    passes = corridor.get("passes", [])
    cand = [(i, _polyline(p)) for i, p in enumerate(passes)]
    cand = [(i, pts) for i, pts in cand if _length(pts) >= min_length_m]
    # longest first: the spine of the stage should win over its stubs
    cand.sort(key=lambda ip: -_length(ip[1]))

    kept, kept_pts = [], []
    for i, pts in cand:
        if _covered_by(pts, kept_pts, dedupe_tol_m) >= dedupe_frac:
            continue                      # a retrace of a road already in the stage
        kept.append(i)
        kept_pts.append(pts)

    # chain: begin at the endpoint furthest from every other endpoint, which is
    # where a stage naturally starts (a dead end, or the edge of the capture)
    remaining = list(zip(kept, kept_pts))
    ends = [(idx, end, pts) for idx, pts in remaining for end in (0, -1)]
    def isolation(item):
        _, end, pts = item
        p = pts[end]
        others = [q[2][q[1]] for q in ends if q[2] is not pts]
        return min((math.dist(p, o) for o in others), default=math.inf)
    start = max(ends, key=isolation)
    order = [{"pass": start[0], "flip": start[1] == -1}]
    chain = _polyline(passes[start[0]], flip=start[1] == -1)
    remaining = [r for r in remaining if r[0] != start[0]]

    while remaining:
        tail = chain[-1]
        best, best_d, best_flip = None, math.inf, False
        for idx, pts in remaining:
            for flip in (False, True):
                head = (pts[::-1] if flip else pts)[0]
                d = math.dist(tail, head)
                if d < best_d:
                    best, best_d, best_flip = idx, d, flip
        if best is None or best_d > join_m:
            break                          # nothing joins on: the stage ends here
        order.append({"pass": best, "flip": best_flip, "gap_m": round(best_d, 2)})
        chain += _polyline(passes[best], flip=best_flip)
        remaining = [r for r in remaining if r[0] != best]

    return {
        "frame": "enu", "origin": corridor["origin"],
        "join_m": join_m, "order": order,
        "length_m": round(_length(chain), 1),
        "dropped_passes": len(passes) - len(order),
        "points": [[round(c, 3) for c in p] for p in chain],
        "start": [round(c, 3) for c in chain[0]],
        "finish": [round(c, 3) for c in chain[-1]],
    }


def build_from_world(world_dir: Path, out: Path | None = None, **kw) -> Path:
    world_dir = Path(world_dir)
    corridor = json.loads((world_dir / "corridor.json").read_text())
    route = build(corridor, **kw)
    path = Path(out or world_dir / "route.json")
    path.write_text(json.dumps(route, indent=1))
    print(f"[route] {len(route['order'])} passes chained, "
          f"{route['length_m'] / 1000:.2f} km stage, "
          f"{route['dropped_passes']} retraces/stubs dropped -> {path}")
    return path


def as_corridor(route: dict) -> dict:
    """Wrap a route as a single-pass corridor so export/guardrail can eat it."""
    return {"frame": "enu", "origin": route["origin"], "radius_m": 25.0,
            "height_band_m": [-1e9, 1e9],
            "passes": [{"video": "route", "points": route["points"]}]}
