# SPDX-License-Identifier: Apache-2.0
"""Keep a camera inside the observed world.

A splat only exists where a camera looked. Free-flight in a corridor-shaped
capture therefore ends in grey void within a few seconds, which reads as "the
reconstruction is broken" when really the viewer just left the data. The fix is
to leash the camera to the capture corridor: project any requested position
onto the nearest observed path and pull it back if it strayed too far.

The same primitive serves the game: the corridor centreline is the drivable
line, `clamp` is the barrier, and `distance` is how far off-road you are.

Pure geometry over `corridor.json` (see corridor.py) -- no renderer, no engine,
so a web viewer, Unity, or a Python notebook can all use it.
"""

import json
import math
from pathlib import Path


def _segments(corridor: dict):
    segs = []
    for pas in corridor.get("passes", []):
        pts = pas["points"]
        segs.extend(zip(pts[:-1], pts[1:]))
    return segs


def _closest_on_segment(p, a, b):
    """Closest point to p on segment ab, plus the squared distance (3D)."""
    ab = [b[i] - a[i] for i in range(3)]
    denom = sum(c * c for c in ab)
    if denom < 1e-12:
        t = 0.0
    else:
        t = sum((p[i] - a[i]) * ab[i] for i in range(3)) / denom
        t = max(0.0, min(1.0, t))
    q = [a[i] + t * ab[i] for i in range(3)]
    return q, sum((p[i] - q[i]) ** 2 for i in range(3))


class Guardrail:
    """Corridor-aware camera limits for one chunk or a whole world."""

    def __init__(self, corridor: dict):
        self.corridor = corridor
        self.segments = _segments(corridor)
        self.radius = float(corridor.get("radius_m", 25.0))
        band = corridor.get("height_band_m") or [-math.inf, math.inf]
        self.height_band = (float(band[0]), float(band[1]))

    @classmethod
    def load(cls, path: Path) -> "Guardrail":
        return cls(json.loads(Path(path).read_text()))

    def nearest(self, pos) -> tuple[list[float], float]:
        """Closest point on any observed path, and the distance to it."""
        if not self.segments:
            return list(pos), 0.0
        best, best_d2 = None, math.inf
        for a, b in self.segments:
            q, d2 = _closest_on_segment(pos, a, b)
            if d2 < best_d2:
                best, best_d2 = q, d2
        return best, math.sqrt(best_d2)

    def distance(self, pos) -> float:
        return self.nearest(pos)[1]

    def inside(self, pos) -> bool:
        return (self.distance(pos) <= self.radius
                and self.height_band[0] <= pos[2] <= self.height_band[1])

    def clamp(self, pos, margin_m: float = 0.0) -> list[float]:
        """Pull `pos` back to the corridor's edge (radius - margin) if outside.

        Slides along the corridor rather than snapping to its centre, so a
        camera pushing sideways at a wall keeps its forward motion instead of
        being yanked onto the driven line.
        """
        limit = max(self.radius - margin_m, 0.0)
        q, d = self.nearest(pos)
        out = list(pos)
        if d > limit > 0:
            scale = limit / d
            out = [q[i] + (pos[i] - q[i]) * scale for i in range(3)]
        lo, hi = self.height_band
        out[2] = min(max(out[2], lo), hi)
        return out


def world_corridor(chunks_dir: Path) -> dict:
    """Union of every chunk's corridor: one leash for the whole world."""
    index = json.loads((Path(chunks_dir) / "chunks.json").read_text())
    passes, radius, lo, hi = [], 0.0, math.inf, -math.inf
    for entry in index["chunks"]:
        path = Path(chunks_dir) / entry["chunk"] / "corridor.json"
        if not path.exists():
            continue
        cor = json.loads(path.read_text())
        radius = max(radius, float(cor.get("radius_m", 0.0)))
        band = cor.get("height_band_m") or [0.0, 0.0]
        lo, hi = min(lo, band[0]), max(hi, band[1])
        for pas in cor.get("passes", []):
            passes.append({**pas, "chunk": entry["chunk"]})
    return {"frame": "enu", "origin": index["origin"],
            "radius_m": radius or 25.0,
            "height_band_m": [lo if lo < math.inf else 0.0,
                              hi if hi > -math.inf else 0.0],
            "passes": passes}
