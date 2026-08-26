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

import json
from pathlib import Path

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
    pts = []
    for f in frames:
        x, y, _z = ll_to_enu(f["lat"], f["lon"], float(f.get("alt") or 0.0),
                             (lat0, lon0, 0.0))
        pts.append((x, y, _z, f))
    return _runs_from_points(pts, gap_m, gap_s)


def _runs_from_points(pts, gap_m, gap_s=30.0):
    """Split (x, y, z, frame) points into passes; see _runs."""
    out, cur, prev = [], [], None
    for pt in pts:
        x, y, _z, f = pt
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
    """Corridor for one chunk's frames. `origin` is the world (lat, lon).

    Positions come from GPS, which is what exists before poses are solved --
    good enough to chunk and to seed spatial matching, and no better than the
    fix was. Once a chunk is solved, `from_sparse` replaces this with the
    camera centres SfM actually recovered (see there for why that matters).
    """
    runs = _runs(frames, origin, gap_m, gap_s)
    out = _assemble(runs, origin, radius_m, height_margin_m, min_spacing_m)
    out["source"] = "gps"
    return out


def from_sparse(chunk: Path, radius_m=25.0, height_margin_m=8.0,
                min_spacing_m=5.0, gap_m=30.0, gap_s=30.0) -> dict | None:
    """Corridor from the SOLVED camera centres in <chunk>/sparse/0.

    The GPS corridor is only as good as the fix. On the first street run the
    recording started in a driveway under canopy with an unconverged
    receiver: the first ~80 positions were 15-19 m off horizontally and 10 m
    off vertically, while SfM was continuous at 0.75 px. A flythrough that
    replays the GPS path there drives through the trees, underground. The
    aligned reconstruction lives in the same ENU frame (model_aligner fitted
    it to the GPS refs where they were good), so its camera centres are the
    honest centreline -- for drive, for pruning, and for the game.

    One point per capture position: the mean of every registered view at that
    seq (they share an optical centre to within the lens baseline). Passes are
    split with the same rules as the GPS corridor. Returns None if the chunk
    has no solved model or no frame records to recover time/video from.
    """
    import pycolmap                                 # noqa: PLC0415

    chunk = Path(chunk)
    sparse = chunk / "sparse" / "0"
    if not (sparse / "images.bin").exists():
        return None
    frames = _chunk_frames(chunk)
    if not frames:
        return None
    rec = pycolmap.Reconstruction(str(sparse))
    centres: dict[int, list] = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        stem = Path(img.name).stem
        if not stem.isdigit():
            continue
        centres.setdefault(int(stem), []).append(img.projection_center())
    if not centres:
        return None
    pts = []
    for f in sorted(frames, key=lambda f: f["seq"]):
        cs = centres.get(f["seq"])
        if not cs:
            continue                    # unregistered position: leave a gap
        x, y, z = (sum(c[i] for c in cs) / len(cs) for i in range(3))
        pts.append((float(x), float(y), float(z), f))
    meta = json.loads((chunk / "meta.json").read_text()) if (chunk / "meta.json").exists() else {}
    origin = ((meta.get("origin") or {}).get("lat"), (meta.get("origin") or {}).get("lon"))
    runs = _runs_from_points(pts, gap_m, gap_s)
    out = _assemble(runs, origin, radius_m, height_margin_m, min_spacing_m)
    out["source"] = "sfm"
    out["registered_positions"] = len(pts)
    out["positions"] = len(frames)
    return out


def _chunk_frames(chunk: Path) -> list[dict]:
    """The chunk's frame records (seq, t, video, lat/lon).

    Chunks written since 2026-08-26 carry their own frames.jsonl; older ones
    are recovered from the ingest directory they were cut from, filtered to
    the images the chunk actually holds.
    """
    own = chunk / "frames.jsonl"
    if own.exists():
        return [json.loads(l) for l in own.read_text().splitlines() if l.strip()]
    parent = chunk.parent.parent / "frames.jsonl"
    if not parent.exists():
        return []
    have = {int(p.stem) for p in (chunk / "images").glob("*/*.jpg") if p.stem.isdigit()}
    return [f for f in (json.loads(l) for l in parent.read_text().splitlines() if l.strip())
            if f["seq"] in have]


def refresh_from_sparse(chunk: Path, **cfg) -> dict | None:
    """Replace <chunk>/corridor.json with the SfM corridor, keeping the GPS
    one as corridor_gps.json. Called by poses after alignment."""
    chunk = Path(chunk)
    cor = from_sparse(chunk, **cfg)
    if cor is None:
        return None
    old = chunk / "corridor.json"
    if old.exists() and json.loads(old.read_text()).get("source", "gps") == "gps":
        old.replace(chunk / "corridor_gps.json")
    old.write_text(json.dumps(cor, indent=1))
    print(f"[corridor] {chunk.name}: rebuilt from SfM -- {len(cor['passes'])} pass(es), "
          f"{cor['registered_positions']}/{cor['positions']} positions registered",
          flush=True)
    return cor


def _assemble(runs, origin, radius_m, height_margin_m, min_spacing_m) -> dict:
    passes = []
    for run in runs:
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
