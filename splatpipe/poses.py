# SPDX-License-Identifier: Apache-2.0
"""Stage 3: per-chunk camera poses via COLMAP (+GLOMAP), geo-aligned to ENU.

Spatial matching uses the GPS priors we wrote into EXIF, which handles
multi-camera rigs and both-direction passes without sequential assumptions.
Work is claimed from a shared queue (see queue.py), so any number of workers
on any number of nodes can be pointed at the same chunk directory.
"""

import os
import shutil
import subprocess
from pathlib import Path

from .queue import WorkQueue


def _run(cmd: list[str], cwd: Path | None = None):
    print("[poses] $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def list_chunks(chunks_dir: Path) -> list[Path]:
    return sorted(p for p in chunks_dir.glob("chunk_*") if (p / "images").is_dir())


def _largest_model(models_dir: Path) -> Path:
    models = [p for p in models_dir.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError(f"no model produced under {models_dir}")
    return max(models, key=lambda p: (p / "images.bin").stat().st_size
               if (p / "images.bin").exists() else 0)


def solve_chunk(chunk: Path, matcher: str = "spatial", align: bool = True,
                use_gpu: bool = True):
    db = chunk / "colmap.db"
    sparse = chunk / "sparse"
    if (sparse / "0").exists():
        print(f"[poses] {chunk.name}: sparse/0 exists, skipping")
        return
    if db.exists():
        db.unlink()  # stale partial runs poison the database; start clean
    sparse.mkdir(exist_ok=True)

    gpu = "1" if use_gpu else "0"
    _run(["colmap", "feature_extractor",
          "--database_path", str(db), "--image_path", str(chunk / "images"),
          "--ImageReader.camera_model", "PINHOLE",
          "--ImageReader.single_camera_per_folder", "1",
          "--SiftExtraction.use_gpu", gpu])

    if matcher == "spatial" and (chunk / "geo.txt").exists():
        _run(["colmap", "spatial_matcher",
              "--database_path", str(db),
              "--SpatialMatching.ignore_z", "1",
              "--SpatialMatching.max_num_neighbors", "32",
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
    print(f"[poses] {chunk.name}: done -> {final}")


def solve_all(chunks_dir: Path, matcher: str = "spatial", align: bool = True):
    chunks = list_chunks(chunks_dir)
    q = WorkQueue(chunks_dir, "poses")
    print(f"[poses] worker {q.worker}: {len(chunks)} chunk(s) in the pool")
    done, failed = q.run(chunks, lambda c: solve_chunk(c, matcher=matcher, align=align))
    print(f"[poses] worker {q.worker}: solved {len(done)}, failed {len(failed)}")
    if failed:
        raise SystemExit(f"[poses] failed chunks: {failed}")
