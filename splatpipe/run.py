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


GATE_FILE = ".seam-gate.json"


def _seam_gate(chunks: Path, fail_over: float | None, out: Path | None = None,
               advisory: bool = False) -> dict:
    """Decide whether this world is worth training, and record the verdict.

    Seam agreement reads corridor.json, which POSES produces -- so a world can
    be judged driveable before a single GPU-hour goes into training. On the
    arrowhead re-bake that is ~2 GPU-hours against ~11, which is the difference
    between iterating on a neighbourhood and not.

    The verdict goes to three places, because three different readers need it
    and only one of them can see the scratch volume:

      <chunks>/.seam-gate.json   the WORKERS. They train too, and a leader that
                                 merely exits would leave them grinding on a
                                 world it has already rejected.
      <out>/seam-gate.json       the SCHEDULER. `out` is the published world
                                 dir, which is the one path a caller is
                                 guaranteed to be able to read -- `chunks`
                                 lives under `work`, which is scratch and which
                                 the world editor cannot mount. Written even
                                 when the gate FAILS and nothing else is
                                 published, which is exactly when it is needed.
      stdout, one line, machine-readable   anyone with only pod logs.

    Written on the skip path as well, so its absence always means something
    went wrong rather than "gate disabled".
    """
    from .seams import measure

    verdict: dict = {"fail_over": fail_over, "advisory": advisory}
    if fail_over is None:
        verdict |= {"ok": True, "skipped": True}
        print("[run] seam gate disabled (--seam-fail-over none)", flush=True)
    else:
        rows, n = measure(chunks)
        if not rows:
            # No overlapping pair means nothing was checked. Passing here would
            # be the "check that cannot fail" that this gate exists to avoid.
            verdict |= {"ok": False, "worst": None, "seams": 0, "chunks": n}
            print(f"[run] seam gate: NOTHING MEASURED -- {n} chunks share no road. "
                  f"Check overlap_m and the corridor radius.", flush=True)
        else:
            worst = max(r[3] for r in rows)
            med = sorted(r[3] for r in rows)[len(rows) // 2]
            # On a global-first world the chunks share one model and this
            # passes by construction (journal 4.1): still worth a line in the
            # log, never a verdict.
            ok = advisory or worst <= fail_over
            verdict |= {"ok": ok, "worst": worst, "median": med,
                        "seams": len(rows), "chunks": n,
                        "offenders": sorted({c for a, b, _, m, _ in rows if m > fail_over
                                             for c in (a, b)})}
            print(f"[run] seam gate: {len(rows)} seams, median {med:.2f} m, "
                  f"worst {worst:.2f} m against a {fail_over:.2f} m bar -- "
                  f"{'ADVISORY (one model, cannot fail)' if advisory else 'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not ok or (advisory and worst > fail_over):
                print(f"[run] chunks over the bar: {verdict['offenders']}", flush=True)
    blob = json.dumps(verdict, indent=1)
    (chunks / GATE_FILE).write_text(blob)
    if out is not None:
        # mkdir here, not at publish time: on a FAILED gate nothing is ever
        # published, and the failure is the thing the caller most needs to read.
        out.mkdir(parents=True, exist_ok=True)
        (out / "seam-gate.json").write_text(blob)
    # One line, prefix-tagged and compact, so a log scraper needs no volume at
    # all: `grep -o 'seam-gate-json .*' | ...`
    print(f"[run] seam-gate-json {json.dumps(verdict, separators=(',', ':'))}", flush=True)
    return verdict


def _wait_for_gate(chunks: Path, timeout_s: float) -> dict:
    """A worker's view of the leader's verdict."""
    t0 = time.time()
    g = chunks / GATE_FILE
    while not g.exists():
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"[run] no {g} after {timeout_s:.0f}s -- did the leader fail?")
        time.sleep(10)
    return json.loads(g.read_text())


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
        timeout_s: float = 48 * 3600,
        seam_fail_over: float | None = 3.0) -> Path:
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

    pcfg = _cfg(config, "poses")
    tcfg = _cfg(config, "train")
    # partition-first solves every chunk on its own and hopes they agree;
    # global-first solves the capture ONCE and cuts it (SCALING-JOURNAL.md,
    # entries 3-5). The order is a property of the world, so it lives in the
    # config next to the mapper, not on the command line.
    order = str(pcfg.get("order", "partition"))
    if order not in ("partition", "global"):
        raise SystemExit(f"[run] poses.order must be partition or global, not {order!r}")
    scene_scale_m = tcfg.get("scene_scale_m")

    if role == "worker":
        chunks = _wait_for_chunks(work, timeout_s)
        if order == "partition":
            solve_all(chunks, matcher=pcfg.get("matcher", "spatial"),
                      align=pcfg.get("align", True),
                      spatial_radius=pcfg.get("spatial_radius", 4),
                      loop_closure=pcfg.get("loop_closure", "none"))
            _wait_for_stage(chunks, "poses", timeout_s)
        # global-first: the leader solves and cuts alone (one model, one pod),
        # and the gate file is the signal that the chunks are ready to train
        if not _wait_for_gate(chunks, timeout_s)["ok"]:
            print("[run] worker: leader rejected this world at the seam gate, "
                  "not training", flush=True)
            return out
        train_all(chunks, scene_scale_m=scene_scale_m)
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

    if order == "global":
        from .globalsfm import solve_global, split_all
        origin = _origin(ccfg.get("origin"))
        if origin is None:
            raise SystemExit("[run] poses.order: global needs chunk.origin in the "
                             "config, or the global model and the cell bounds "
                             "will not share a frame")
        # One solve over the whole capture (solve_global refuses a partial one:
        # under GLOBAL_MIN_COVERAGE of the frames registered is a hole, not a
        # world), then every cell cut from it. Nothing here is claim-based --
        # a global solve is one process by nature -- so the workers wait on
        # the gate file rather than the poses queue.
        solve_global(frames, origin,
                     matcher=pcfg.get("matcher", "spatial"),
                     align=pcfg.get("align", True),
                     spatial_radius=pcfg.get("spatial_radius", 4),
                     mapper=pcfg.get("mapper", "auto"),
                     loop_closure=pcfg.get("loop_closure", "none"))
        split_all(frames, chunks, rule=pcfg.get("rule", "inria"),
                  refine=bool(pcfg.get("refine", False)))
        # The seam check passes by construction on a global-first world (the
        # chunks share one model: entry 4.1), so it is reported, never a bar.
        # The bar that matters was solve_global's coverage, already applied.
        gate = _seam_gate(chunks, seam_fail_over, out=out, advisory=True)
    else:
        # the leader is also a worker: on a one-pod run this is the whole
        # pipeline, and on a JobSet it just means the leader's GPU is not idle
        # while the workers grind
        solve_all(chunks, matcher=pcfg.get("matcher", "spatial"), align=pcfg.get("align", True),
                  spatial_radius=pcfg.get("spatial_radius", 4),
                  loop_closure=pcfg.get("loop_closure", "none"))
        _wait_for_stage(chunks, "poses", timeout_s)
        gate = _seam_gate(chunks, seam_fail_over, out=out)
    if not gate["ok"]:
        # Non-zero, with the number on stdout: a scheduler's run object then
        # fails visibly instead of publishing a world nobody trusts.
        worst = gate.get("worst")
        raise SystemExit(f"[run] seam gate FAILED: worst seam "
                         f"{'none measured' if worst is None else format(worst, '.2f') + ' m'} "
                         f"against a {seam_fail_over:.2f} m bar. Not training. "
                         f"Verdict in {chunks / GATE_FILE}.")
    train_all(chunks, scene_scale_m=scene_scale_m)
    _wait_for_stage(chunks, "train", timeout_s)

    out.mkdir(parents=True, exist_ok=True)
    merge(chunks, out, prune_corridor=True)
    for name, keep in (("mid", 0.40), ("probe", 0.125)):
        build_lod(out, keep=keep, sh=False, name=name)
    print(f"[run] world published to {out}", flush=True)
    return out
