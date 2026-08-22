# SPDX-License-Identifier: Apache-2.0
"""Stage 2: split the geotagged frame stream into overlapping spatial chunks.

Each chunk is a self-contained COLMAP workspace: images/ (symlinks into the
ingest output) + geo.txt subset + meta.json. chunk_m == 0 -> one chunk with
everything (smoke tests, GPS-less data).
"""

import json
import os
from pathlib import Path

from .geo import track_distances


def _load_frames(frames_dir: Path) -> list[dict]:
    with open(frames_dir / "frames.jsonl") as fh:
        return [json.loads(line) for line in fh]


def _write_chunk(frames_dir: Path, chunk_dir: Path, members: list[dict]):
    images = chunk_dir / "images"
    geo_lines = []
    geo_src = {}
    geo_path = frames_dir / "geo.txt"
    if geo_path.exists():
        for line in geo_path.read_text().splitlines():
            name, rest = line.split(" ", 1)
            geo_src[name] = rest
    for frame in members:
        for rel in frame["images"].values():
            dst = images / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = frames_dir / "images" / rel
            if not dst.exists():
                os.symlink(os.path.relpath(src, dst.parent), dst)
            if rel in geo_src:
                geo_lines.append(f"{rel} {geo_src[rel]}")
    if geo_lines:
        (chunk_dir / "geo.txt").write_text("\n".join(geo_lines) + "\n")
    (chunk_dir / "meta.json").write_text(json.dumps(
        {"frames": len(members), "seq_range": [members[0]["seq"], members[-1]["seq"]]}))


def make_chunks(frames_dir: Path, chunk_m: float = 200.0, overlap_m: float = 40.0,
                min_frames: int = 20) -> Path:
    frames = _load_frames(frames_dir)
    chunks_dir = frames_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    have_gps = frames and frames[0].get("lat") is not None
    if not have_gps or chunk_m <= 0:
        groups = [frames]
    else:
        dist = track_distances([f["lat"] for f in frames], [f["lon"] for f in frames])
        stride = max(chunk_m - overlap_m, 1.0)
        groups = []
        start = 0.0
        while start < dist[-1]:
            members = [f for f, d in zip(frames, dist) if start <= d < start + chunk_m]
            if len(members) >= min_frames:
                groups.append(members)
            start += stride

    manifest = []
    for ci, members in enumerate(groups):
        if not members:
            continue
        chunk_dir = chunks_dir / f"chunk_{ci:03d}"
        chunk_dir.mkdir(exist_ok=True)
        _write_chunk(frames_dir, chunk_dir, members)
        manifest.append({"chunk": chunk_dir.name, "frames": len(members)})
    (chunks_dir / "chunks.json").write_text(json.dumps(manifest, indent=2))
    print(f"[chunk] {len(manifest)} chunks -> {chunks_dir}")
    return chunks_dir
