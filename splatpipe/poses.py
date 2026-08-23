# SPDX-License-Identifier: Apache-2.0
"""Stage 3: per-chunk camera poses via COLMAP (+GLOMAP), geo-aligned to ENU.

spatial matching uses the GPS priors we wrote into EXIF, which handles
multi-camera rigs and both-direction passes without sequential assumptions.
Rank-sharded: under torchrun/devpod each rank solves chunks[RANK::WORLD_SIZE].
"""

import os
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None):
    print("[poses] $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def rank_shard(items: list) -> list:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return items[rank::world]


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
    if align and (chunk / "geo.txt").exists():
        final.mkdir(exist_ok=True)
        _run(["colmap", "model_aligner",
              "--input_path", str(model), "--output_path", str(final),
              "--ref_images_path", str(chunk / "geo.txt"),
              "--ref_is_gps", "1", "--alignment_type", "enu",
              "--alignment_max_error", "3"])
    else:
        shutil.move(str(model), str(final))
    print(f"[poses] {chunk.name}: done -> {final}")


def solve_all(chunks_dir: Path, matcher: str = "spatial", align: bool = True):
    mine = rank_shard(list_chunks(chunks_dir))
    print(f"[poses] rank {os.environ.get('RANK', 0)}: {len(mine)} chunk(s)")
    failed = []
    for chunk in mine:
        try:
            solve_chunk(chunk, matcher=matcher, align=align)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"[poses] {chunk.name}: FAILED ({e})")
            failed.append(chunk.name)
    if failed:
        raise SystemExit(f"[poses] failed chunks: {failed}")
