# SPDX-License-Identifier: Apache-2.0
"""Stage 4: per-chunk gaussian splat training, fanned out over a work queue.

IMPORTANT: we train with --no-normalize-world-space. gsplat's trainer defaults
to recentring and rescaling the scene into a unit-ish box, which silently
undoes the shared ENU frame that poses.py went to trouble to establish: chunk
gaussians would land in a different per-chunk frame, so merge could not simply
concatenate, corridor pruning would match nothing, and drive/mesh would render
an empty world (observed: a splat spanning +-2 units against a corridor
spanning hundreds of metres).

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
from .workqueue import WorkQueue


def train_chunk(chunk: Path, examples: Path, steps: int, extra: list[str],
                preset: str = "default", tag: str | None = None):
    # `tag` keeps experiments side by side: splat_<tag>/ next to splat/, and
    # its own queue stage, so "one change at a time" comparisons never
    # overwrite the run they are being compared against
    result = chunk / (f"splat_{tag}" if tag else "splat")
    if list(result.glob("**/*.ply")):
        print(f"[train] {chunk.name}: ply exists, skipping")
        return
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LOCAL_RANK", "0"))
    # route through our wrapper when masks exist, so the rig is excluded from
    # the loss instead of being fitted as phantom geometry
    masked = (chunk / "masks").is_dir()
    entry = (["-m", "splatpipe.gsplat_masked", str(examples)]
             if masked else [str(examples / "simple_trainer.py")])
    cmd = [sys.executable, *entry, preset,
           "--data-dir", str(chunk), "--data-factor", "1",
           "--result-dir", str(result), "--max-steps", str(steps),
           "--save-ply", "--disable-viewer",
           "--no-normalize-world-space",   # keep gaussians in the project ENU frame
           # Anti-aliasing matters when the same surface is seen from 2 m and
           # from 60 m in one chunk, which is every driving capture.
           "--antialiased",
           *extra]
    print("[train] $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"[train] {chunk.name}: done -> {result}")


def train_all(chunks_dir: Path, steps: int = 30000, extra: list[str] | None = None,
              only: list[str] | None = None, preset: str = "default",
              tag: str | None = None):
    examples = Path(os.environ.get("GSPLAT_EXAMPLES", "/opt/gsplat/examples"))
    if not (examples / "simple_trainer.py").exists():
        raise SystemExit(f"gsplat examples not found at {examples} (set GSPLAT_EXAMPLES)")
    ready = [c for c in list_chunks(chunks_dir) if (c / "sparse" / "0").exists()]
    if only:
        ready = [c for c in ready if any(o in c.name for o in only)]
        print(f"[train] --only {only}: {len(ready)} chunk(s)")
    q = WorkQueue(chunks_dir, f"train_{tag}" if tag else "train")
    print(f"[train] worker {q.worker}: {len(ready)} posed chunk(s) in the pool"
          + (f" (preset {preset}, tag {tag})" if tag or preset != "default" else ""))
    done, failed = q.run(ready, lambda c: train_chunk(c, examples, steps, extra or [],
                                                      preset=preset, tag=tag))
    print(f"[train] worker {q.worker}: trained {len(done)}, failed {len(failed)}")
    if failed:
        raise SystemExit(f"[train] failed chunks: {failed}")
