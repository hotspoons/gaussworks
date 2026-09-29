# SPDX-License-Identifier: Apache-2.0
"""Solve the whole capture once, THEN cut it up.

Why this module exists. Until now the pipeline partitioned first and solved
second: each chunk ran its own COLMAP/GLOMAP and then aligned itself to its own
GPS priors with `model_aligner`. Every chunk therefore chose its datum alone,
and nothing in the pipeline required two chunks covering the same road to agree
about its height. Measured over two bakes, ~44% of chunks ended up more than
3 m from a neighbour, and that rate did not move when the cell size changed --
because cell size changes how MANY independent datums there are, not the odds
of any one of them being wrong (docs/SCALING-JOURNAL.md, entries 2.4a and 3).

Hierarchical 3DGS (Kerbl et al., SIGGRAPH 2024), built for this exact regime,
orders it the other way: global SfM over the entire capture, align that single
reconstruction to metric, and only then cut it into chunks, which inherit their
poses from the global solve. Partitioning is a TRAINING optimisation applied to
a world that is already consistent. Their chunks cannot disagree about a shared
road because the road's height was decided once, before the cut.

That is what this module does.

    write_enu_reference(frames, origin)   all frames' GPS -> the project ENU frame
    solve_global(frames, ...)             one reconstruction for the whole capture
    split_to_chunks(frames, chunks, ...)  crop it into the existing chunk dirs

A note on cost, because it reads as if it must be more expensive and is not.
The survey solved 10,035 frame-memberships to cover 5,075 frames -- every 40 m
halo is reconstructed twice, once for each chunk that shares it. A global solve
touches each frame once. It is roughly HALF the work, and the halves it drops
are the redundant ones.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from .geo import ll_to_enu

GLOBAL_SPARSE = "sparse/0"


def write_enu_reference(frames_dir: Path, origin: tuple[float, float]) -> Path:
    """Every frame's GPS fix, in the project ENU frame, for `model_aligner`.

    `chunk` writes one of these per chunk; the global solve needs the same file
    over all frames and against the SAME pinned origin, or the global model
    lands in a different world from the chunk bounds we are about to cut with.
    """
    out = frames_dir / "geo_enu.txt"
    lines = []
    with (frames_dir / "frames.jsonl").open() as fh:
        for raw in fh:
            frame = json.loads(raw)
            if frame.get("lat") is None:
                continue
            e, n, u = ll_to_enu(frame["lat"], frame["lon"],
                                float(frame.get("alt") or 0.0),
                                (origin[0], origin[1], 0.0))
            for rel in frame["images"].values():
                lines.append(f"{rel} {e:.4f} {n:.4f} {u:.4f}")
    if not lines:
        raise SystemExit(f"[global] no frame in {frames_dir} has a GPS fix; "
                         "the global solve has nothing to align to")
    out.write_text("\n".join(lines) + "\n")
    print(f"[global] {len(lines)} ENU reference positions -> {out}", flush=True)
    return out


def solve_global(frames_dir: Path, origin: tuple[float, float], **kw) -> Path:
    """One reconstruction over every frame in the capture.

    The frames dir already has the shape `solve_chunk` expects -- images/,
    masks/, cameras.json -- so this is the same solver over a bigger set rather
    than a second implementation that could drift from it.
    """
    from .poses import solve_chunk               # noqa: PLC0415  (circular)

    write_enu_reference(frames_dir, origin)
    solve_chunk(frames_dir, **kw)
    model = frames_dir / GLOBAL_SPARSE
    if not model.exists():
        raise SystemExit(f"[global] solver produced no {model}")

    # The POINT of a global solve is that one model covers the capture. If it
    # does not, every downstream stage still "works" -- the split writes empty
    # chunks, training skips them, merge produces a world with holes, and the
    # seam check reads fine because the chunks it can still compare are the ones
    # that survived. Refuse here, where it is one line, rather than let that
    # play out over a day.
    import pycolmap                              # noqa: PLC0415
    n_ref = len((frames_dir / "geo_enu.txt").read_text().splitlines())
    n_reg = pycolmap.Reconstruction(str(model)).num_reg_images()
    frac = n_reg / n_ref if n_ref else 0.0
    print(f"[global] {n_reg:,} of {n_ref:,} images registered ({frac:.0%})", flush=True)
    if frac < float(os.environ.get("GLOBAL_MIN_COVERAGE", "0.85")):
        raise SystemExit(
            f"[global] only {frac:.0%} of the capture is in the global model. "
            f"A partial solve is worse than none: it publishes a world with "
            f"holes and nothing downstream complains. Check the mapper's model "
            f"count above -- if it fragmented, the leaves did not merge, and a "
            f"different mapper (or more matching) is the fix, not a lower bar.")
    return model


def _bbox(meta: dict, halo_m: float) -> str:
    """The chunk's cell, grown by its halo, as a model_cropper boundary.

    Altitude is left wide open. The bounds are a ground-plan grid; clipping in
    z would cut the tops off buildings and the canopy, and on a hill it would
    cut the road itself.
    """
    e0, n0, e1, n1 = meta["bounds_enu_m"]
    e0, e1 = min(e0, e1) - halo_m, max(e0, e1) + halo_m
    n0, n1 = min(n0, n1) - halo_m, max(n0, n1) + halo_m
    return f"{e0},{n0},-100000,{e1},{n1},100000"


def _relink(frames_dir: Path, chunk: Path, names: set[str]) -> int:
    """Make the chunk's images/ and masks/ cover exactly what its model cites.

    `model_cropper` keeps the POINTS inside the box and then any image that
    observes them, so the cropped model legitimately cites cameras standing
    outside the cell and looking in -- which is what we want (it is also what
    the hierarchical-3DGS chunker does, deliberately). But the trainer opens
    every image the model names, so any of those not already symlinked here is
    a FileNotFoundError halfway through a chunk.
    """
    added = 0
    for rel in sorted(names):
        dst = chunk / "images" / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = frames_dir / "images" / rel
            if not src.exists():
                raise SystemExit(f"[global] {chunk.name}: model cites {rel}, "
                                 f"which is not in {frames_dir / 'images'}")
            # relative, matching chunks.py: the PVC is mounted at different
            # paths in different pods, and an absolute link breaks on the next
            # one to open it
            os.symlink(os.path.relpath(src, dst.parent), dst)
            added += 1
        msrc = frames_dir / "masks" / (rel + ".png")
        if msrc.exists():
            mdst = chunk / "masks" / (rel + ".png")
            if not mdst.exists():
                mdst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(os.path.relpath(msrc.resolve(), mdst.parent), mdst)
    return added


def split_chunk(frames_dir: Path, chunk: Path, halo_m: float | None = None,
                global_sparse: Path | None = None) -> dict:
    """Cut one chunk's sub-model out of the global reconstruction."""
    import pycolmap                              # noqa: PLC0415

    model = global_sparse or (frames_dir / GLOBAL_SPARSE)
    meta = json.loads((chunk / "meta.json").read_text())
    halo = float(meta.get("overlap_m", 40.0) if halo_m is None else halo_m)
    final = chunk / "sparse" / "0"
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True, exist_ok=True)

    subprocess.run(["colmap", "model_cropper",
                    "--input_path", str(model), "--output_path", str(final),
                    "--boundary", _bbox(meta, halo)], check=True)

    rec = pycolmap.Reconstruction(str(final))
    names = {im.name for im in rec.images.values()}
    added = _relink(frames_dir, chunk, names)
    stats = {"chunk": chunk.name, "images": rec.num_reg_images(),
             "points": rec.num_points3D(), "linked": added}
    if stats["images"] < 2:
        # Not a warning. A chunk with nothing in it trains into an empty tile
        # and merges into a hole in the world, silently.
        print(f"[global] {chunk.name}: ONLY {stats['images']} images in the "
              f"crop -- this cell is empty or the bounds are wrong", flush=True)
    print(f"[global] {chunk.name}: {stats['images']} images, "
          f"{stats['points']} points, {added} new links", flush=True)
    try:
        from . import corridor as corridor_mod    # noqa: PLC0415
        corridor_mod.refresh_from_sparse(chunk)
    except Exception as exc:                      # noqa: BLE001
        print(f"[global] {chunk.name}: corridor refresh skipped ({exc})", flush=True)
    return stats
