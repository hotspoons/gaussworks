# SPDX-License-Identifier: Apache-2.0
"""Stage 3: per-chunk camera poses via COLMAP (+GLOMAP), geo-aligned to ENU.

Spatial matching uses the GPS priors we wrote into EXIF, which handles
multi-camera rigs and both-direction passes without sequential assumptions.
Work is claimed from a shared queue (see queue.py), so any number of workers
on any number of nodes can be pointed at the same chunk directory.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

from .workqueue import WorkQueue


def _run(cmd: list[str], cwd: Path | None = None):
    # flush: COLMAP writes straight to the fd, so unflushed python prints land
    # in the log long after the subprocess output they were meant to label
    print("[poses] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def list_chunks(chunks_dir: Path) -> list[Path]:
    return sorted(p for p in chunks_dir.glob("chunk_*") if (p / "images").is_dir())


def _largest_model(models_dir: Path) -> Path:
    models = [p for p in models_dir.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError(f"no model produced under {models_dir}")
    return max(models, key=lambda p: (p / "images.bin").stat().st_size
               if (p / "images.bin").exists() else 0)


def _rig_size(chunk: Path) -> tuple[int, int]:
    """(cameras, passes) — how many images share one capture position."""
    cams_path, meta_path = chunk / "cameras.json", chunk / "meta.json"
    n_cams = len(json.loads(cams_path.read_text())) if cams_path.exists() else 1
    n_pass = 1
    if meta_path.exists():
        n_pass = max(1, len(json.loads(meta_path.read_text()).get("passes") or []))
    return n_cams, n_pass


def solve_chunk(chunk: Path, matcher: str = "spatial", align: bool = True,
                use_gpu: bool = True, spatial_radius: int = 4):
    db = chunk / "colmap.db"
    sparse = chunk / "sparse"
    if (sparse / "0").exists():
        print(f"[poses] {chunk.name}: sparse/0 exists, skipping", flush=True)
        return
    if db.exists():
        db.unlink()  # stale partial runs poison the database; start clean
    sparse.mkdir(exist_ok=True)

    gpu = "1" if use_gpu else "0"
    extract = ["colmap", "feature_extractor",
               "--database_path", str(db), "--image_path", str(chunk / "images"),
               "--ImageReader.camera_model", "PINHOLE",
               "--ImageReader.single_camera_per_folder", "1",
               "--SiftExtraction.use_gpu", gpu]
    cams_path = chunk / "cameras.json"
    if cams_path.exists():
        cams = json.loads(cams_path.read_text())
        c = cams[0]
        key = ("fx", "fy", "cx", "cy", "width", "height")
        # camera_params is a single string applied to every folder, so it is
        # only safe when the view plan gave every camera the same intrinsics --
        # which auto planning does. A hand-written mixed-FOV view list would
        # silently get the first view's focal length imposed on all of them.
        if all(tuple(x[k] for k in key) == tuple(c[k] for k in key) for x in cams):
            extract += ["--ImageReader.camera_params",
                        f"{c['fx']:.6f},{c['fy']:.6f},{c['cx']:.6f},{c['cy']:.6f}"]
            print(f"[poses] intrinsics from cameras.json: fx={c['fx']:.1f} "
                  f"cx={c['cx']:.1f} ({c['width']}x{c['height']}, {c['fov']} deg)",
                  flush=True)
        else:
            print("[poses] WARNING: cameras.json holds mixed intrinsics, so they "
                  "cannot be pinned per folder; COLMAP will guess fx = 1.2*max(w,h) "
                  "and registration will suffer. Use one FOV/size across views.",
                  flush=True)
    if (chunk / "masks").is_dir():
        # keeps features off the capture vehicle, which is rigid in the camera
        # frame and would otherwise drag every pose toward itself
        extract += ["--ImageReader.mask_path", str(chunk / "masks")]
    _run(extract)

    if matcher == "spatial" and (chunk / "geo.txt").exists():
        n_cams, n_pass = _rig_size(chunk)
        # SIZE THE NEIGHBOURHOOD TO THE RIG, or spatial matching quietly
        # collapses. Every virtual view of one capture position carries that
        # position's GPS, and every pass down the road revisits it, so
        # n_cams * n_pass images sit at essentially the same coordinate --
        # 6 x 3 = 18 here. A fixed 32 neighbours is then +-1 position of road,
        # the match graph becomes a razor-thin chain, and the incremental
        # mapper builds ONE local component and stops: observed as a
        # contiguous block of 100 of 331 positions registered, every camera at
        # the identical rate.
        neighbors = max(32, n_cams * n_pass * spatial_radius * 2)
        print(f"[poses] {chunk.name}: {n_cams} cams x {n_pass} pass(es) at each "
              f"position -> {neighbors} spatial neighbours "
              f"(~{spatial_radius} positions either side)", flush=True)
        _run(["colmap", "spatial_matcher",
              "--database_path", str(db),
              "--SpatialMatching.ignore_z", "1",
              "--SpatialMatching.max_num_neighbors", str(neighbors),
              "--SiftMatching.use_gpu", gpu])
        # Then the long chain along the road. Image names are
        # camK/NNNNNN.jpg, so name order is per-camera and per-pass in capture
        # order: sequential matching adds exactly the reach spatial matching
        # cannot afford, at a fraction of the pairs. Matches accumulate in the
        # same database, so this is additive.
        _run(["colmap", "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", "15",
              "--SiftMatching.use_gpu", gpu])
    elif matcher == "exhaustive":
        # small chunks: all-pairs matching links the rig's cam folders, which
        # sequential (name-ordered) matching never crosses
        _run(["colmap", "exhaustive_matcher",
              "--database_path", str(db),
              "--SiftMatching.use_gpu", gpu])
    else:
        _run(["colmap", "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", "15",
              "--SiftMatching.use_gpu", gpu])

    raw = sparse / "raw"
    raw.mkdir(exist_ok=True)
    if shutil.which("glomap"):
        _run(["glomap", "mapper", "--database_path", str(db),
              "--image_path", str(chunk / "images"), "--output_path", str(raw)])
    else:
        _run(["colmap", "mapper", "--database_path", str(db),
              "--image_path", str(chunk / "images"), "--output_path", str(raw)])
    model = _largest_model(raw)

    final = sparse / "0"
    enu_ref, gps_ref = chunk / "geo_enu.txt", chunk / "geo.txt"
    if align and (enu_ref.exists() or gps_ref.exists()):
        final.mkdir(exist_ok=True)
        if enu_ref.exists():
            # positions already in the project ENU frame -> every chunk lands in
            # the same world, which is what makes merging a concatenation
            ref_args = ["--ref_images_path", str(enu_ref), "--ref_is_gps", "0",
                        "--alignment_type", "custom"]
        else:
            ref_args = ["--ref_images_path", str(gps_ref), "--ref_is_gps", "1",
                        "--alignment_type", "enu"]
        _run(["colmap", "model_aligner",
              "--input_path", str(model), "--output_path", str(final),
              *ref_args, "--alignment_max_error", "3"])
    else:
        shutil.move(str(model), str(final))
    print(f"[poses] {chunk.name}: done -> {final}", flush=True)


def solve_all(chunks_dir: Path, matcher: str = "spatial", align: bool = True,
              only: list[str] | None = None, spatial_radius: int = 4):
    chunks = list_chunks(chunks_dir)
    if only:
        chunks = [c for c in chunks if any(o in c.name for o in only)]
        print(f"[poses] --only {only}: {len(chunks)} chunk(s)", flush=True)
    q = WorkQueue(chunks_dir, "poses")
    print(f"[poses] worker {q.worker}: {len(chunks)} chunk(s) in the pool", flush=True)
    done, failed = q.run(chunks, lambda c: solve_chunk(
        c, matcher=matcher, align=align, spatial_radius=spatial_radius))
    print(f"[poses] worker {q.worker}: solved {len(done)}, failed {len(failed)}", flush=True)
    if failed:
        raise SystemExit(f"[poses] failed chunks: {failed}")
