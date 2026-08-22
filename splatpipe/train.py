# SPDX-License-Identifier: Apache-2.0
"""Stage 4: per-chunk gaussian splat training, sharded across ranks.

Chunks are independent, so there are no collectives: each rank runs the gsplat
reference trainer on chunks[RANK::WORLD_SIZE], pinned to LOCAL_RANK's GPU.
Under `devpod launch` a 4-node x 4-GPU group trains 16 chunks concurrently;
plain single-GPU invocation works identically (RANK=0, WORLD_SIZE=1).
"""

import os
import subprocess
import sys
from pathlib import Path

from .poses import list_chunks, rank_shard


def train_chunk(chunk: Path, examples: Path, steps: int, extra: list[str]):
    result = chunk / "splat"
    if list(result.glob("**/*.ply")):
        print(f"[train] {chunk.name}: ply exists, skipping")
        return
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LOCAL_RANK", "0"))
    cmd = [sys.executable, str(examples / "simple_trainer.py"), "default",
           "--data-dir", str(chunk), "--data-factor", "1",
           "--result-dir", str(result), "--max-steps", str(steps),
           "--save-ply", "--disable-viewer", *extra]
    print("[train] $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"[train] {chunk.name}: done -> {result}")


def train_all(chunks_dir: Path, steps: int = 30000, extra: list[str] | None = None):
    examples = Path(os.environ.get("GSPLAT_EXAMPLES", "/opt/gsplat/examples"))
    if not (examples / "simple_trainer.py").exists():
        raise SystemExit(f"gsplat examples not found at {examples} (set GSPLAT_EXAMPLES)")
    ready = [c for c in list_chunks(chunks_dir) if (c / "sparse" / "0").exists()]
    mine = rank_shard(ready)
    print(f"[train] rank {os.environ.get('RANK', 0)}: {len(mine)} of {len(ready)} chunk(s)")
    failed = []
    for chunk in mine:
        try:
            train_chunk(chunk, examples, steps, extra or [])
        except subprocess.CalledProcessError as e:
            print(f"[train] {chunk.name}: FAILED ({e})")
            failed.append(chunk.name)
    if failed:
        raise SystemExit(f"[train] failed chunks: {failed}")
