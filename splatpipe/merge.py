# SPDX-License-Identifier: Apache-2.0
"""Stage 5: stitch per-chunk splats into one streamable world.

Because every chunk was posed against the same project ENU frame (see
chunks.py / poses.py), its gaussians are already world-space: there is nothing
to transform here, and in particular no spherical-harmonic rotation, which is
what makes naive splat merging lossy. Merging is therefore selection:

1. OWNERSHIP -- a chunk contributes only the gaussians inside its own cell
   bounds. The halo exists to improve that chunk's solve, not to donate
   geometry, so the neighbour that owns a region is the one that reconstructed
   it with full surrounding coverage. Half-open bounds mean no gaussian is
   claimed twice and none is dropped.
2. CORRIDOR PRUNE -- gaussians farther than radius_m from any observed camera
   path (or outside the height band) are floaters: reconstruction noise in
   space no camera ever saw. This is the cheapest big win in both size and
   visual quality.
3. TILES -- one PLY per cell plus world.json, so an engine streams by locality
   instead of loading a monolith. --single also writes the concatenation.
"""

import json
from pathlib import Path

import numpy as np

from . import ply


def _segments(corridor: dict) -> tuple[np.ndarray, np.ndarray]:
    """Corridor polylines as (start, end) point pairs in XY."""
    a, b = [], []
    for pas in corridor.get("passes", []):
        pts = np.asarray(pas["points"], dtype=np.float64)[:, :2]
        if len(pts) >= 2:
            a.append(pts[:-1])
            b.append(pts[1:])
    if not a:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.concatenate(a), np.concatenate(b)


def _corridor_mask(pts: np.ndarray, corridor: dict) -> np.ndarray:
    """Keep gaussians within radius of a camera path and inside the height band.

    Distance is measured to the polyline SEGMENTS, not to its vertices: the
    paths are decimated, so vertex-only distance would carve scallops out of
    the corridor between points and prune perfectly well-observed geometry.
    """
    seg_a, seg_b = _segments(corridor)
    if not len(seg_a):
        return np.ones(len(pts), bool)
    radius = float(corridor.get("radius_m", 25.0))
    lo, hi = corridor.get("height_band_m", [-1e9, 1e9])

    keep = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
    ab = seg_b - seg_a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    idx = np.flatnonzero(keep)
    for start in range(0, len(idx), 20_000):          # blocked: N x S x 2 floats
        sl = idx[start:start + 20_000]
        d = pts[sl, None, :2] - seg_a[None, :, :]
        t = np.clip((d * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
        diff = d - t[..., None] * ab[None]
        keep[sl] = (diff * diff).sum(-1).min(axis=1) <= radius * radius
    return keep


def _chunk_ply(chunk: Path) -> Path | None:
    plys = sorted((chunk / "splat").rglob("*.ply"))
    if not plys:
        return None
    # highest training step wins: point_cloud_29999.ply over point_cloud_6999.ply
    def step(p):
        digits = "".join(c for c in p.stem if c.isdigit())
        return int(digits) if digits else -1
    return max(plys, key=step)


def merge(chunks_dir: Path, out: Path, prune_corridor: bool = True,
          single: bool = False) -> Path:
    index = json.loads((chunks_dir / "chunks.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    tiles_dir = out / "tiles"

    tiles, kept_total, raw_total = [], 0, 0
    for entry in index["chunks"]:
        chunk = chunks_dir / entry["chunk"]
        src = _chunk_ply(chunk)
        if src is None:
            print(f"[merge] {entry['chunk']}: no ply, skipping")
            continue
        meta = json.loads((chunk / "meta.json").read_text())
        data, _ = ply.read(src)
        pts = ply.xyz(data)
        raw_total += len(data)

        x0, y0, x1, y1 = meta["bounds_enu_m"]
        keep = ((pts[:, 0] >= x0) & (pts[:, 0] < x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] < y1))
        owned = int(keep.sum())

        corridor_path = chunk / "corridor.json"
        if prune_corridor and corridor_path.exists():
            corridor = json.loads(corridor_path.read_text())
            keep &= _corridor_mask(pts, corridor)

        sel = data[keep]
        kept_total += len(sel)
        tile = tiles_dir / f"{entry['chunk']}.ply"
        ply.write(tile, sel)
        tiles.append({"tile": tile.name, "chunk": entry["chunk"],
                      "bounds_enu_m": meta["bounds_enu_m"],
                      "centre": meta.get("centre"),
                      "gaussians": len(sel), "owned": owned, "source": len(data)})
        print(f"[merge] {entry['chunk']}: {len(data)} -> owned {owned} "
              f"-> kept {len(sel)}")

    world = {"frame": "enu", "origin": index["origin"], "cell_m": index["cell_m"],
             "gaussians": kept_total, "source_gaussians": raw_total,
             "corridor_pruned": prune_corridor, "tiles": tiles}
    (out / "world.json").write_text(json.dumps(world, indent=1))

    if single and tiles:
        parts = [ply.read(tiles_dir / t["tile"])[0] for t in tiles]
        ply.write(out / "world.ply", np.concatenate(parts))

    print(f"[merge] {len(tiles)} tiles, {kept_total} gaussians "
          f"({raw_total} before ownership/prune) -> {out}")
    return out
