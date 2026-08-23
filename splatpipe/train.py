# SPDX-License-Identifier: Apache-2.0
"""Stage 4: per-chunk gaussian splat training, fanned out over a work queue.

Chunks are independent, so there are no collectives: each worker claims a chunk
from the shared queue and runs the gsplat reference trainer on it, pinned to
LOCAL_RANK's GPU. Under `devpod launch` a 4-node x 4-GPU group trains 16 chunks
at a time and rebalances automatically; a single GPU works identically.
"""

import os
import subprocess
import sys
from pathlib import Path

from .poses import list_chunks
from .queue import WorkQueue


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
    q = WorkQueue(chunks_dir, "train")
    print(f"[train] worker {q.worker}: {len(ready)} posed chunk(s) in the pool")
    done, failed = q.run(ready, lambda c: train_chunk(c, examples, steps, extra or []))
    print(f"[train] worker {q.worker}: trained {len(done)}, failed {len(failed)}")
    if failed:
        raise SystemExit(f"[train] failed chunks: {failed}")
