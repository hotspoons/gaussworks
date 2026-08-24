# SPDX-License-Identifier: Apache-2.0
"""Stage 2: split a geotagged frame stream into overlapping spatial chunks.

Chunks are LOCALITY buckets, not slices of the drive. A fixed grid is laid over
the capture area in local ENU metres, and every frame whose position falls in a
cell joins that cell's chunk -- whichever pass, whichever direction. Driving a
road twice therefore strengthens one reconstruction instead of producing two
competing ones, and an intersection gets observations from every approach.

Each cell also pulls in frames from an `overlap_m` halo beyond its bounds, so
neighbouring chunks share observations and align cleanly when merged.

Output per chunk (a self-contained COLMAP workspace):
    images/camN/*.jpg   symlinks into the ingest output
    geo.txt             GPS priors for this chunk's images
    corridor.json       observed envelope (see corridor.py)
    meta.json           cell bounds, centre, frame/pass counts
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path

from . import corridor as corridor_mod
from .geo import ll_to_enu


def _load_frames(frames_dir: Path) -> list[dict]:
    with open(frames_dir / "frames.jsonl") as fh:
        return [json.loads(line) for line in fh]


def _cells_for(v: float, cell_m: float, overlap_m: float) -> set[int]:
    """Cell indices along one axis whose (expanded) span contains v."""
    return {math.floor((v - overlap_m) / cell_m), math.floor(v / cell_m),
            math.floor((v + overlap_m) / cell_m)}


def _write_chunk(frames_dir: Path, chunk_dir: Path, members: list[dict],
                 geo_src: dict, origin: tuple, meta: dict, corridor_cfg: dict):
    images = chunk_dir / "images"
    geo_lines, enu_lines = [], []
    for frame in members:
        enu = None
        if frame.get("lat") is not None:
            enu = ll_to_enu(frame["lat"], frame["lon"], float(frame.get("alt") or 0.0),
                            (origin[0], origin[1], 0.0))
        for rel in frame["images"].values():
            dst = images / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.symlink(os.path.relpath(frames_dir / "images" / rel, dst.parent), dst)
            src_mask = frames_dir / "masks" / (rel + ".png")
            if src_mask.exists():
                mdst = chunk_dir / "masks" / (rel + ".png")
                mdst.parent.mkdir(parents=True, exist_ok=True)
                if not mdst.exists():
                    os.symlink(os.path.relpath(src_mask.resolve(), mdst.parent), mdst)
            if rel in geo_src:
                geo_lines.append(f"{rel} {geo_src[rel]}")
            if enu is not None:
                enu_lines.append(f"{rel} {enu[0]:.4f} {enu[1]:.4f} {enu[2]:.4f}")
    if geo_lines:
        (chunk_dir / "geo.txt").write_text("\n".join(geo_lines) + "\n")
    # Reference positions in the ONE project frame. COLMAP's own --alignment_type
    # enu would centre each chunk on its own GPS centroid, leaving every chunk in
    # a different frame; aligning to these instead means all chunks (and their
    # gaussians) share a world, so merge is concatenation.
    if enu_lines:
        (chunk_dir / "geo_enu.txt").write_text("\n".join(enu_lines) + "\n")
    if members and members[0].get("lat") is not None:
        (chunk_dir / "corridor.json").write_text(json.dumps(
            corridor_mod.build(members, origin, **corridor_cfg), indent=1))
    (chunk_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def _geo_index(frames_dir: Path) -> dict:
    path = frames_dir / "geo.txt"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        name, rest = line.split(" ", 1)
        out[name] = rest
    return out


def make_chunks(frames_dir: Path, cell_m: float = 200.0, overlap_m: float = 40.0,
                min_frames: int = 20, corridor_cfg: dict | None = None) -> Path:
    frames = _load_frames(frames_dir)
    chunks_dir = frames_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    geo_src = _geo_index(frames_dir)
    corridor_cfg = corridor_cfg or {}

    have_gps = bool(frames) and frames[0].get("lat") is not None
    origin = (frames[0]["lat"], frames[0]["lon"]) if have_gps else (0.0, 0.0)

    if not have_gps or cell_m <= 0:
        groups = {(0, 0): (frames, len(frames))}
    else:
        # One pass over the frames: each lands in its own cell (core) and in
        # any neighbour whose halo reaches it. Cheap regardless of area.
        core = defaultdict(list)
        members = defaultdict(list)
        for f in frames:
            x, y, _ = ll_to_enu(f["lat"], f["lon"], float(f.get("alt") or 0.0),
                                (origin[0], origin[1], 0.0))
            cx, cy = math.floor(x / cell_m), math.floor(y / cell_m)
            core[(cx, cy)].append(f)
            for ix in _cells_for(x, cell_m, overlap_m):
                for iy in _cells_for(y, cell_m, overlap_m):
                    members[(ix, iy)].append(f)
        # A cell earns a chunk on its own observations, not on borrowed halo.
        groups = {c: (members[c], len(core[c])) for c in core
                  if len(core[c]) >= min_frames}

    manifest = []
    for (cx, cy), (mem, n_core) in sorted(groups.items()):
        mem = sorted(mem, key=lambda f: f["seq"])
        name = f"chunk_x{cx}_y{cy}" if have_gps and cell_m > 0 else "chunk_000"
        chunk_dir = chunks_dir / name
        chunk_dir.mkdir(exist_ok=True)
        passes = sorted({f.get("video") for f in mem if f.get("video")})
        centre = None
        if have_gps:
            centre = {"lat": sum(f["lat"] for f in mem) / len(mem),
                      "lon": sum(f["lon"] for f in mem) / len(mem)}
        meta = {"cell": [cx, cy], "cell_m": cell_m, "overlap_m": overlap_m,
                "frames": len(mem), "core_frames": n_core, "centre": centre,
                "origin": {"lat": origin[0], "lon": origin[1]},
                "bounds_enu_m": [cx * cell_m, cy * cell_m,
                                 (cx + 1) * cell_m, (cy + 1) * cell_m],
                "passes": passes}
        _write_chunk(frames_dir, chunk_dir, mem, geo_src, origin, meta, corridor_cfg)
        manifest.append({"chunk": name, "frames": len(mem),
                         "core_frames": n_core, "passes": len(passes)})

    (chunks_dir / "chunks.json").write_text(json.dumps(
        {"origin": {"lat": origin[0], "lon": origin[1]}, "cell_m": cell_m,
         "overlap_m": overlap_m, "chunks": manifest}, indent=1))
    total = sum(c["frames"] for c in manifest)
    print(f"[chunk] {len(manifest)} chunks, {total} frame-memberships "
          f"({len(frames)} frames, cell={cell_m}m overlap={overlap_m}m) -> {chunks_dir}")
    return chunks_dir
