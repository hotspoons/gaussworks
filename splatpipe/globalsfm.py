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


def _select_cameras(rec, meta: dict, halo_m: float, rule: str) -> set[int]:
    """Which of the cropped model's cameras this chunk should actually train on.

    This is the knob that decides how much of its budget a tile spends on ground
    it does not own. `model_cropper` keeps every image observing a point in the
    box, which reaches a long way down a straight road: measured on gosheff, a
    chunk trained ~300k gaussians and merge kept ~24k of them, because the other
    92% fell in a neighbour's cell. A tile optimising mostly other people's
    ground is a tile spending its budget badly, and global-first came out ~1 dB
    below partition-first on exactly those chunks.

      crop   every image model_cropper returned (what we shipped first)
      cell   only cameras standing inside the cell plus its halo
      inria  cameras inside the cell, OR within 2x the cell that also SEE at
             least 50 points inside it -- the rule hierarchical-3DGS uses, and
             the one this module should have copied along with the cropping
    """
    import numpy as np                           # noqa: PLC0415

    e0, n0, e1, n1 = meta["bounds_enu_m"]
    e0, e1 = min(e0, e1), max(e0, e1)
    n0, n1 = min(n0, n1), max(n0, n1)
    if rule == "crop":
        return set(rec.images.keys())

    inner = (e0 - halo_m, n0 - halo_m, e1 + halo_m, n1 + halo_m)
    # 2x the cell about its centre, per the reference rule
    ce, cn = (e0 + e1) / 2, (n0 + n1) / 2
    we, wn = (e1 - e0), (n1 - n0)
    outer = (ce - we, cn - wn, ce + we, cn + wn)

    def inside(box, p):
        return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]

    keep = set()
    for iid, im in rec.images.items():
        c = np.asarray(im.projection_center())
        if inside(inner, c):
            keep.add(iid)
        elif rule == "inria" and inside(outer, c):
            seen = 0
            for p2 in im.points2D:
                if p2.has_point3D():
                    xyz = rec.points3D[p2.point3D_id].xyz
                    if inside((e0, n0, e1, n1), xyz):
                        seen += 1
                        if seen >= 50:
                            break
            if seen >= 50:
                keep.add(iid)
    return keep


def reprojection_rms(rec, max_points: int = 40000) -> float:
    """Root-mean-square reprojection error, measured, in pixels.

    Not `compute_mean_reprojection_error`: that averages the error field
    stored on each point, and model_cropper writes cropped points with that
    field at zero, so on a cut model the stored figure reads 0.000 before AND
    after anything is done to it. A check that cannot fail is not a check.
    """
    import numpy as np                           # noqa: PLC0415

    pids = list(rec.points3D.keys())
    step = max(1, len(pids) // max_points)
    sq = []
    for pid in pids[::step]:
        pt = rec.points3D[pid]
        for el in pt.track.elements:
            im = rec.images[el.image_id]
            p = im.cam_from_world * pt.xyz
            if p[2] <= 0:
                continue
            uv = rec.cameras[im.camera_id].img_from_cam(p[:2] / p[2])
            xy = im.points2D[el.point2D_idx].xy
            sq.append(float(np.sum((np.asarray(uv) - xy) ** 2)))
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def refine_chunk(final: Path, max_move_m: float = 0.5) -> dict:
    """Bundle-adjust the cut model locally, the step hierarchical-3DGS takes
    after ITS cut and the one this module left out.

    The global solve decides where every camera is to within its own
    tolerance -- one bundle adjustment over 30k images, three passes. A tile
    trains against those poses as given. Refining the cut model lets the
    ~1-2k cameras of one cell settle against the points they actually see,
    the way a partition-first chunk's final BA did. Intrinsics stay fixed:
    the virtual pinholes are exact by construction (SEAM.md), and letting
    each cell re-estimate them is how neighbours start to disagree again.

    The crop also clips observations: a camera at the halo edge looking
    outward keeps only the few points that fell inside the box, and a local
    BA lets a camera that weakly constrained wander. Measured on gosheff,
    the median camera moved 3-9 cm and one moved 5.4 m. So any camera that
    moves more than `max_move_m` gets its global pose back: for that camera
    the whole-capture solve is the better-conditioned estimate, and a 5 m
    step at a cell edge is exactly the seam this ordering exists to remove.
    """
    import numpy as np                           # noqa: PLC0415
    import pycolmap                              # noqa: PLC0415

    before_rec = pycolmap.Reconstruction(str(final))
    before = reprojection_rms(before_rec)
    was = {iid: (im.cam_from_world, np.asarray(im.projection_center()))
           for iid, im in before_rec.images.items()}
    subprocess.run(["colmap", "bundle_adjuster",
                    "--input_path", str(final), "--output_path", str(final),
                    "--BundleAdjustment.refine_focal_length", "0",
                    "--BundleAdjustment.refine_principal_point", "0",
                    "--BundleAdjustment.refine_extra_params", "0",
                    "--BundleAdjustment.refine_extrinsics", "1",
                    "--BundleAdjustment.max_num_iterations", "50"], check=True)
    rec = pycolmap.Reconstruction(str(final))
    moved = np.array([np.linalg.norm(np.asarray(im.projection_center()) - was[iid][1])
                      for iid, im in rec.images.items()])
    reverted = 0
    for iid, im in rec.images.items():
        if np.linalg.norm(np.asarray(im.projection_center()) - was[iid][1]) > max_move_m:
            im.cam_from_world = was[iid][0]
            reverted += 1
    if reverted:
        rec.write(str(final))
    after = reprojection_rms(rec)
    return {"rms_before_px": before, "rms_after_px": after,
            "moved_median_m": float(np.median(moved)) if len(moved) else 0.0,
            "moved_max_m": float(moved.max()) if len(moved) else 0.0,
            "reverted": reverted}


def relocated_frames(frames_dir: Path, chunk: Path,
                     global_sparse: Path | None = None) -> tuple[int, int]:
    """(frames this cell was dealt by GPS, how many the global model registered).

    A cell can come out of the cut empty for two reasons that need opposite
    responses. If its frames are not in the global model at all, the solve
    has a hole and the world would too: stop. If they ARE registered -- just
    not inside this cell -- then the GPS that dealt them here was wrong and
    the model corrected it (the first 45 s of arrowhead sit 200 m from their
    cold-start fix), so the cell was never real: drop it and carry on.
    """
    import pycolmap                              # noqa: PLC0415

    listed = chunk / "geo_enu.txt"
    names = [ln.split()[0] for ln in listed.read_text().splitlines() if ln.strip()] \
        if listed.exists() else []
    rec = pycolmap.Reconstruction(str(global_sparse or (frames_dir / GLOBAL_SPARSE)))
    have = {im.name for im in rec.images.values()}
    return len(names), sum(1 for n in names if n in have)


SEEN_REACH_M = 300.0


def _keep_seen_points(model: Path, final: Path, meta: dict, halo_m: float,
                      keep_names: set[str]) -> None:
    """Rebuild the cut with the chosen cameras and every point two of them
    observe within SEEN_REACH_M of the cell, wherever it lies.

    model_cropper keeps only points inside the box. A partition-first chunk
    was never boxed: it holds whatever its cameras saw, and 6-12% of its
    points sit outside the cell+halo (median 17 m out, p90 60-100 m) -- the
    houses, trees and road beyond the halo that every view looking outward
    is scored against. A cut that drops them starts training the background
    from nothing.

    Done with the cropper and a wide box rather than by loading the global
    model in Python: eight of those loads of the neighbourhood model (4.5M
    points, 30k images) OOM-killed a 128 GiB pod. The cropper is C++ and
    streams; 300 m of reach covers the p90 with room to spare.
    """
    import pycolmap                              # noqa: PLC0415

    shutil.rmtree(final)
    final.mkdir(parents=True)
    subprocess.run(["colmap", "model_cropper",
                    "--input_path", str(model), "--output_path", str(final),
                    "--boundary", _bbox(meta, halo_m + SEEN_REACH_M)], check=True)
    rec = pycolmap.Reconstruction(str(final))
    # COLMAP's DeRegisterImage drops each observation and deletes any point
    # whose track falls below two, so what is left is exactly the points the
    # kept cameras still triangulate between them.
    for iid, im in list(rec.images.items()):
        if im.name not in keep_names:
            rec.deregister_image(iid)
    rec.write(str(final))


def _retriangulate(frames_dir: Path, final: Path) -> None:
    """Triangulate the cut afresh from the capture's own matches, poses fixed.

    The global solve's tracks are what one mapper made over 30k images; a
    partition-first chunk's final model carried ~17% more observations per
    image on the same cameras. point_triangulator re-derives points from the
    database for the registered images only (it filters the cache by image
    name, so the 41 GB neighbourhood database is not read whole), leaves the
    extrinsics alone -- no seam can open -- and refines nothing but points.
    """
    subprocess.run(["colmap", "point_triangulator",
                    "--database_path", str(frames_dir / "colmap.db"),
                    "--image_path", str(frames_dir / "images"),
                    "--input_path", str(final), "--output_path", str(final),
                    "--clear_points", "1",
                    "--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0"], check=True)


def split_chunk(frames_dir: Path, chunk: Path, halo_m: float | None = None,
                global_sparse: Path | None = None, rule: str = "crop",
                refine: bool = False, points: str = "box") -> dict:
    """Cut one chunk's sub-model out of the global reconstruction.

    `points` decides which 3D points the cut keeps: `box` = inside the
    cell+halo (model_cropper), `seen` = everything the kept cameras observe,
    `retri` = seen, then re-triangulated from the database with poses fixed.
    """
    import pycolmap                              # noqa: PLC0415

    if points not in ("box", "seen", "retri"):
        raise SystemExit(f"[global] points must be box, seen or retri, not {points!r}")
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
    keep = _select_cameras(rec, meta, halo, rule)
    if rule != "crop":
        dropped = [i for i in list(rec.images.keys()) if i not in keep]
        for iid in dropped:
            rec.deregister_image(iid)
        for pid in [p for p, pt in rec.points3D.items() if pt.track.length() < 2]:
            rec.delete_point3D(pid)
        rec.write(str(final))
        print(f"[global] {chunk.name}: rule={rule} kept {len(keep)} of "
              f"{len(keep) + len(dropped)} cameras", flush=True)
        rec = pycolmap.Reconstruction(str(final))
    if points != "box":
        boxed = rec.num_points3D()
        _keep_seen_points(model, final, meta, halo, {rec.images[i].name for i in keep})
        if points == "retri":
            _retriangulate(frames_dir, final)
        rec = pycolmap.Reconstruction(str(final))
        print(f"[global] {chunk.name}: points={points} {boxed} in the box -> "
              f"{rec.num_points3D()} kept", flush=True)
    if refine and rec.num_reg_images() >= 2:
        r = refine_chunk(final)
        print(f"[global] {chunk.name}: local BA, reprojection RMS "
              f"{r['rms_before_px']:.3f} -> {r['rms_after_px']:.3f} px; cameras moved "
              f"median {r['moved_median_m'] * 100:.1f} cm, max {r['moved_max_m']:.2f} m; "
              f"{r['reverted']} put back to their global pose", flush=True)
        rec = pycolmap.Reconstruction(str(final))
    names = {im.name for im in rec.images.values()}
    added = _relink(frames_dir, chunk, names)
    stats = {"chunk": chunk.name, "rule": rule, "points_rule": points,
             "images": rec.num_reg_images(), "points": rec.num_points3D(),
             "linked": added, "refined": bool(refine)}
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


def split_all(frames_dir: Path, chunks_dir: Path, rule: str = "crop",
              refine: bool = False, halo_m: float | None = None,
              global_sparse: Path | None = None,
              only: list[str] | None = None, points: str = "box") -> list[dict]:
    """Cut every chunk out of the global model; park cells that the solve
    emptied by putting their GPS-dealt frames elsewhere; refuse on a hole.

    `only` narrows to chunks whose name contains one of the substrings, so a
    49-cell split with --refine (minutes of BA per cell) can be fanned out
    over processes; each invocation is independent of the others.
    """
    from .poses import list_chunks                # noqa: PLC0415

    chunks = list_chunks(chunks_dir)
    if only:
        chunks = [c for c in chunks if any(o in c.name for o in only)]
    if not chunks:
        raise SystemExit(f"[global] no chunks under {chunks_dir}"
                         + (f" matching {only}" if only else ""))
    rows = [split_chunk(frames_dir, c, halo_m=halo_m, global_sparse=global_sparse,
                        rule=rule, refine=refine, points=points) for c in chunks]
    print(f"[global] split {len(rows)} chunks; "
          f"{sum(r['images'] for r in rows)} image memberships, "
          f"{sum(r['linked'] for r in rows)} new links", flush=True)
    holes = []
    for name in [r["chunk"] for r in rows if r["images"] < 2]:
        chunk = chunks_dir / name
        dealt, registered = relocated_frames(frames_dir, chunk, global_sparse)
        if dealt and registered >= 0.9 * dealt:
            # the GPS dealt these frames to a cell the solve says they are
            # not in; the model has them, in the neighbours' cuts
            parked = chunks_dir / "relocated" / name
            parked.parent.mkdir(exist_ok=True)
            if parked.exists():
                shutil.rmtree(parked)
            shutil.move(str(chunk), str(parked))
            print(f"[global] {name}: empty because its {dealt} frames are "
                  f"registered elsewhere ({registered} in the global model): "
                  f"a GPS-dealt cell, not a hole. Parked at {parked}", flush=True)
        else:
            holes.append((name, dealt, registered))
    if holes:
        raise SystemExit(f"[global] {len(holes)} chunk(s) came out empty with "
                         f"frames the global model never registered: {holes}")
    return rows


def refine_with_priors(frames_dir: Path, out: Path | None = None,
                       sigma_xy_m: float = 12.0, sigma_z_m: float = 20.0,
                       max_iterations: int = 40, threads: int = 0) -> Path:
    """Bundle-adjust the global model with GPS as a WEAK position prior.

    Why this exists. A whole-capture solve with no GPS in the loop is locally
    right and globally free: on the neighbourhood it registered every image
    in one model, cut with seams of 0.02 m -- and drifted off the map. Against
    the OSM road layer the GPS track sits a median 2.4 m from a road over the
    whole drive (p90 13.9 m); the solve sits at p90 44.9 m, and on one 580-
    frame stretch runs 37 m off-road where the GPS runs 1.6 m. An out-and-
    back spur with no cross-links came out foreshortened by 350 m. A single
    similarity alignment at the end (model_aligner) cannot remove
    low-frequency drift, and the seam check cannot see it (journal 4.2).

    Partition-first never had this problem because every chunk was aligned
    to ITS OWN GPS -- which is also why the chunks disagreed. This is the
    middle: one model, so neighbours cannot disagree, with the GPS pulling at
    ~sigma so that the low-frequency shape follows the map while the image
    constraints keep the local geometry rigid. Sigma is the GPS error under
    canopy here (DOP up to 19: 12 m horizontal, worse vertically), and the
    loss is robust so the cold-start fixes pull nothing.

    Intrinsics stay fixed (exact virtual pinholes). Output goes beside the
    input, never over it, so the two can be compared.
    """
    import numpy as np                           # noqa: PLC0415
    import pycolmap                              # noqa: PLC0415

    model = frames_dir / GLOBAL_SPARSE
    out = out or (frames_dir / "sparse" / "prior")
    rec = pycolmap.Reconstruction(str(model))
    enu = {}
    for ln in (frames_dir / "geo_enu.txt").read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 4:
            enu[parts[0]] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    cov = np.diag([sigma_xy_m ** 2, sigma_xy_m ** 2, sigma_z_m ** 2])
    priors = {}
    for iid, im in rec.images.items():
        if im.name in enu:
            pp = pycolmap.PosePrior()
            pp.position = enu[im.name]
            pp.position_covariance = cov
            pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            priors[iid] = pp
    print(f"[global] priors for {len(priors):,} of {rec.num_reg_images():,} images "
          f"(sigma {sigma_xy_m:.0f} m horizontal, {sigma_z_m:.0f} m vertical)", flush=True)

    before = _prior_residual(rec, enu)
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = False
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    opts.refine_extrinsics = True
    opts.print_summary = True
    opts.solver_options.max_num_iterations = int(max_iterations)
    opts.solver_options.num_threads = int(threads) if threads else max(1, (os.cpu_count() or 2) - 2)
    popts = pycolmap.PosePriorBundleAdjustmentOptions()
    popts.use_robust_loss_on_prior_position = True
    popts.ransac_max_error = 0.0          # already in ENU; do not re-align
    config = pycolmap.BundleAdjustmentConfig()
    for iid in rec.images:
        config.add_image(iid)
    ba = pycolmap.create_pose_prior_bundle_adjuster(opts, popts, config, priors, rec)
    ba.solve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec.write(str(out))
    after = _prior_residual(rec, enu)
    print(f"[global] |solve - GPS| median {before[0]:.1f} -> {after[0]:.1f} m, "
          f"p90 {before[1]:.1f} -> {after[1]:.1f} m, max {before[2]:.1f} -> {after[2]:.1f} m; "
          f"model written to {out}", flush=True)
    return out


def _prior_residual(rec, enu: dict) -> tuple[float, float, float]:
    import numpy as np                           # noqa: PLC0415

    d = np.array([np.linalg.norm(np.asarray(im.projection_center())[:2] - enu[im.name][:2])
                  for im in rec.images.values() if im.name in enu])
    return (float(np.median(d)), float(np.percentile(d, 90)), float(d.max())) if len(d) else (0., 0., 0.)
