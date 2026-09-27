# SPDX-License-Identifier: Apache-2.0
"""One entrypoint for a whole capture, so a scheduler never has to know the stages.

    splatpipe run --capture /data/captures/<id> --out /data/splats/<id> --role leader
    splatpipe run --capture /data/captures/<id> --out /data/splats/<id> --role worker

WHY A ROLE AND NOT A SCRIPT PER STAGE. The pipeline is two shapes, not one. Ingest,
mask, chunk, merge and the LOD levels are SERIAL and happen once; poses and train are
CLAIM-BASED and any number of workers can pull from them (workqueue.py). A JobSet can
express exactly that with a leader job and N worker jobs, and it does not need to know
which stage is which -- it only needs the two roles.

THE COUNT OF WORKERS DOES NOT HAVE TO BE RIGHT. A worker that finds an empty queue
finishes; a worker that starts before the chunks exist waits for them. So parallelism can
be set from free GPUs rather than from the footage, and over-provisioning costs an idle
pod rather than a wrong answer. That is a property of the claim queue, and it is the
reason this is safe to hand to a scheduler.

WHAT THE LEADER NEEDS THAT A WORKER DOES NOT: the video (workers never read it), enough
disk for the frames, and a GPU only for NVDEC during ingest. Everything after chunk is
CPU. A worker needs one GPU and no video at all.
"""

import json
import os
import time
from pathlib import Path


def _chapters(capture: Path) -> tuple[str, list[Path]]:
    """The camera to ingest and its chapters, in capture order.

    A capture may carry several cameras. They are separate rigs with separate
    optical centres, and `ingest` refuses to mix profiles for that reason -- so
    this takes ONE camera per run and says which, rather than silently ingesting
    the first and dropping the rest.
    """
    man = json.loads((capture / "capture.json").read_text())
    cams = man.get("cameras") or {}
    if not cams:
        raise SystemExit(f"[run] {capture}/capture.json lists no cameras")
    want = os.environ.get("CAPTURE_CAMERA")
    if want and want not in cams:
        raise SystemExit(f"[run] CAPTURE_CAMERA={want!r} is not in {sorted(cams)}")
    name = want or sorted(cams)[0]
    if not want and len(cams) > 1:
        print(f"[run] {len(cams)} cameras in this capture; ingesting {name!r}. "
              f"Set CAPTURE_CAMERA to pick another: {sorted(cams)}", flush=True)
    files = [capture / "video" / name / c["name"] if "path" not in c else Path(c["path"])
             for c in cams[name]]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"[run] chapters missing on disk: {[str(m) for m in missing]}")
    return name, files


def _queue_state(chunks: Path, stage: str) -> tuple[int, int, int]:
    q = chunks / ".queue" / stage
    names = [p.name for p in chunks.glob("chunk_*") if (p / "images").is_dir()]
    done = sum(1 for n in names if (q / f"{n}.done").exists())
    running = sum(1 for n in names if (q / f"{n}.lock").is_dir())
    return done, running, len(names)


def _wait_for_chunks(work: Path, timeout_s: float) -> Path:
    chunks = work / "frames" / "chunks"
    index = chunks / "chunks.json"
    t0 = time.time()
    said = False
    while not index.exists():
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"[run] no {index} after {timeout_s:.0f}s -- did the leader fail?")
        if not said:
            print(f"[run] waiting for the leader to publish {index}", flush=True)
            said = True
        time.sleep(10)
    return chunks


def _wait_for_stage(chunks: Path, stage: str, timeout_s: float):
    """Block until every chunk has a .done for this stage."""
    t0 = time.time()
    last = None
    while True:
        done, running, total = _queue_state(chunks, stage)
        if total and done >= total:
            print(f"[run] {stage}: {done}/{total} complete", flush=True)
            return
        if (done, running) != last:
            print(f"[run] {stage}: {done}/{total} done, {running} running", flush=True)
            last = (done, running)
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"[run] {stage} stuck at {done}/{total} after {timeout_s:.0f}s")
        time.sleep(20)


def run(capture: Path, out: Path, role: str, config: str | None,
        site: Path | None = None, work: Path | None = None,
        timeout_s: float = 48 * 3600) -> Path:
    from .chunks import make_chunks
    from .cli import _cfg, _near, _origin
    from .mask import build as build_masks
    from .merge import merge
    from .poses import solve_all
    from .train import train_all
    from .lod import build as build_lod
    from .ingest import ingest_videos

    work = work or (out / ".work")
    work.mkdir(parents=True, exist_ok=True)

    if role == "worker":
        chunks = _wait_for_chunks(work, timeout_s)
        pcfg = _cfg(config, "poses")
        solve_all(chunks, matcher=pcfg.get("matcher", "spatial"),
                  align=pcfg.get("align", True),
                  spatial_radius=pcfg.get("spatial_radius", 4),
                  loop_closure=pcfg.get("loop_closure", "none"))
        _wait_for_stage(chunks, "poses", timeout_s)
        train_all(chunks)
        print("[run] worker done", flush=True)
        return out

    # ---- leader ---------------------------------------------------------------
    camera, files = _chapters(capture)
    print(f"[run] capture {capture.name}: camera {camera}, {len(files)} chapter(s)", flush=True)
    frames = work / "frames"
    icfg = _cfg(config, "ingest")
    view_cfg = {k: v for k, v in icfg.items()
                if k in ("fov", "px_per_deg", "pitch", "aspect", "view_overlap_deg", "min_coverage")}
    if not (frames / "frames.jsonl").exists():
        ingest_videos(files, frames, view_cfg=view_cfg,
                      extract_fps=icfg.get("extract_fps", 6.0),
                      spacing_m=icfg.get("spacing_m", 1.75),
                      jpeg_quality=icfg.get("jpeg_quality", 95),
                      hwaccel=icfg.get("hwaccel", "cuda"),
                      jobs=icfg.get("jobs", 0),
                      near=_near(icfg.get("near")),
                      radius_m=icfg.get("radius_m", 400.0))
    else:
        print(f"[run] {frames}/frames.jsonl exists, skipping ingest", flush=True)
    build_masks(frames)
    ccfg = _cfg(config, "chunk")
    chunks = make_chunks(frames, cell_m=ccfg.get("cell_m", 200.0),
                         overlap_m=ccfg.get("overlap_m", 40.0),
                         min_frames=ccfg.get("min_frames", 20),
                         corridor_cfg=_cfg(config, "corridor"),
                         origin=_origin(ccfg.get("origin")))

    # the leader is also a worker: on a one-pod run this is the whole pipeline, and on a
    # JobSet it just means the leader's GPU is not idle while the workers grind
    pcfg = _cfg(config, "poses")
    solve_all(chunks, matcher=pcfg.get("matcher", "spatial"), align=pcfg.get("align", True),
              spatial_radius=pcfg.get("spatial_radius", 4),
              loop_closure=pcfg.get("loop_closure", "none"))
    _wait_for_stage(chunks, "poses", timeout_s)
    train_all(chunks)
    _wait_for_stage(chunks, "train", timeout_s)

    out.mkdir(parents=True, exist_ok=True)
    merge(chunks, out, prune_corridor=True)
    for name, keep in (("mid", 0.40), ("probe", 0.125)):
        build_lod(out, keep=keep, sh=False, name=name)
    print(f"[run] world published to {out}", flush=True)
    return out
