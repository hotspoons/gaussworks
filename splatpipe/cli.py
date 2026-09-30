# SPDX-License-Identifier: Apache-2.0
"""splatpipe CLI: ingest -> chunk -> poses -> train."""

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def _near(v) -> tuple[float, float] | None:
    """Accept "lat,lon" or [lat, lon]; None means no filter."""
    if v is None:
        return None
    if isinstance(v, str):
        return tuple(float(x) for x in v.split(","))  # type: ignore[return-value]
    return (float(v[0]), float(v[1]))


def _origin(v) -> tuple[float, float] | None:
    """Accept "lat,lon" from the CLI, or {lat, lon} / [lat, lon] from a config."""
    if v is None:
        return None
    if isinstance(v, str):
        lat, lon = (float(x) for x in v.split(","))
        return (lat, lon)
    if isinstance(v, dict):
        return (float(v["lat"]), float(v["lon"]))
    return (float(v[0]), float(v[1]))


def _cfg(path: str | None, stage: str) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get(stage, {}) or {}


def main():
    p = argparse.ArgumentParser(prog="splatpipe")
    p.add_argument("--config", help="stage-defaults yaml (see configs/)")

    # --config is a GLOBAL flag, so `splatpipe run --config x` was an
    # "unrecognized arguments" exit 2 -- a whole job dying on its first line
    # because a flag was on the wrong side of the subcommand. Anyone generating
    # this command line hits it, and the world editor did. Accepting it in
    # either position costs one parent parser.
    #
    # SUPPRESS is what makes it safe: without it the subparser's default of
    # None would overwrite a --config given BEFORE the subcommand, turning the
    # working spelling into a silent no-config run.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help="stage-defaults yaml (see configs/)")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(parents=[common], name="ingest", help="video(s) -> geotagged pinhole frames")
    s.add_argument("videos", nargs="+", type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--profile", help="camera profile name (default: auto-detect)")
    s.add_argument("--fov", type=float, help="virtual view FOV, degrees")
    s.add_argument("--px-per-deg", type=float, help="angular resolution of the views")
    s.add_argument("--pitch", help="comma-separated view pitches, degrees (down = negative)")
    s.add_argument("--extract-fps", type=float)
    s.add_argument("--spacing-m", type=float)
    s.add_argument("--hwaccel", help="ffmpeg decoder, e.g. cuda (8K HEVC is decode-bound)")
    s.add_argument("--jobs", type=int, help="threads rendering virtual views "
                                            "(default: min(32, cores))")
    s.add_argument("--start-s", type=float, default=0.0, help="skip into the clip")
    s.add_argument("--duration-s", type=float, help="ingest only this many seconds")
    s.add_argument("--near", help="lat,lon: keep only frames near this point")
    s.add_argument("--radius-m", type=float, default=400.0)
    s.add_argument("--segment-s", type=float, default=60.0,
                   help="extract in chunks this long; bounds temp disk use")
    s.add_argument("--no-telemetry", action="store_true",
                   help="proceed without GPS (no geo alignment, no locality "
                        "chunking, no distance-based spacing)")

    s = sub.add_parser(parents=[common], name="mapillary", help="fetch 360 sequences w/ GPS from Mapillary")
    s.add_argument("--bbox", required=True, help="w,s,e,n")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--max-images", type=int, default=2000)
    s.add_argument("--min-seq-len", type=int, default=50)

    s = sub.add_parser(parents=[common], name="chunk", help="frames -> overlapping locality chunks")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--cell-m", type=float, help="grid cell size, metres (0 = one chunk)")
    s.add_argument("--overlap-m", type=float, help="halo pulled in from neighbours")
    s.add_argument("--min-frames", type=int, help="cells with fewer own frames are dropped")
    s.add_argument("--origin", help="lat,lon of the project ENU origin and cell "
                                    "grid. Default: the first frame's fix, which "
                                    "is usually the least converged one in the drive")

    s = sub.add_parser(parents=[common], name="profiles", help="list camera profiles, or detect one for a file")
    s.add_argument("video", nargs="?", type=Path)
    s.add_argument("--plan", action="store_true", help="also show the view plan")

    s = sub.add_parser(parents=[common], name="flatten", help=".360 -> equirectangular mp4 for any 360 player")
    s.add_argument("video", type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--width", type=int, default=4096)
    s.add_argument("--fps", type=float, help="defaults to the source rate")
    s.add_argument("--start-s", type=float, default=0.0)
    s.add_argument("--duration-s", type=float)
    s.add_argument("--hwaccel", help="e.g. cuda")
    s.add_argument("--profile")

    s = sub.add_parser(parents=[common], name="mask", help="auto-mask the capture vehicle out of every frame")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--sample", type=int, default=60)
    s.add_argument("--search-from", type=float, default=0.35,
                   help="fraction down the frame where the rig may start")
    s.add_argument("--dark-pct", type=float, default=45.0,
                   help="percentile of median luminance treated as rig")
    s.add_argument("--seam-band-deg", type=float, default=0.0,
                   help="pre-stitched input only: mask this many degrees either "
                        "side of the seam. Raw-lens profiles never need it -- "
                        "ingest writes exact per-lens coverage masks instead.")

    s = sub.add_parser(parents=[common], name="status", help="queue state across chunks (pending/running/done/failed)")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--stage", default="all", choices=["all", "poses", "train"])

    s = sub.add_parser(parents=[common], name="poses", help="per-chunk COLMAP/GLOMAP + ENU alignment")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--matcher", choices=["spatial", "sequential", "exhaustive"])
    s.add_argument("--no-align", action="store_true")
    s.add_argument("--only", nargs="*", help="chunk name substrings: run just these")
    s.add_argument("--mapper", choices=["auto", "glomap", "colmap"],
                   default="auto",
                   help="auto = glomap if installed. Pin it when the two "
                        "disagree; see the version-skew note in poses.py")
    s.add_argument("--refresh", action="store_true",
                   help="re-extract features and re-match, instead of reusing "
                        "a complete database (the default)")
    s.add_argument("--spatial-radius", type=int, default=4,
                   help="spatial matching reach, in capture positions either "
                        "side (scaled internally by cameras x passes). GPS "
                        "error larger than this reach hides cross-pass pairs; "
                        "see --loop-closure")
    s.add_argument("--loop-closure", choices=["none", "vocab"], default=None,
                   help="add retrieval-based matching (COLMAP vocab tree) after "
                        "spatial+sequential, for multi-pass chunks whose GPS "
                        "priors cannot be trusted (default: config, else none)")

    s = sub.add_parser(parents=[common], name="global-solve",
                       help="ONE reconstruction for the whole capture, before chunking")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--matcher", default="spatial")
    s.add_argument("--spatial-radius", type=int)
    s.add_argument("--mapper", default="auto",
                   choices=["auto", "glomap", "colmap", "hierarchical"],
                   help="hierarchical = COLMAP's divide-and-conquer over ONE "
                        "database, for captures too large to solve monolithically")
    s.add_argument("--loop-closure", default=None)
    s.add_argument("--origin", help="lat,lon of the project ENU frame; defaults to "
                                    "the chunk config's origin so the global model "
                                    "lands in the same world as the cell bounds")

    s = sub.add_parser(parents=[common], name="global-split",
                       help="cut a global reconstruction into the existing chunks")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--sparse", type=Path, help="default: <frames>/sparse/0")
    s.add_argument("--halo-m", type=float, help="default: each chunk's own overlap_m")
    s.add_argument("--rule", default="crop", choices=["crop", "cell", "inria"],
                   help="which cameras a chunk trains on. crop = everything "
                        "model_cropper returned; cell = only cameras inside the "
                        "cell+halo; inria = inside, or within 2x the cell and "
                        "seeing 50+ points in it (hierarchical-3DGS's rule)")
    s.add_argument("--refine", action="store_true",
                   help="bundle-adjust each cut model locally (extrinsics + points, "
                        "intrinsics fixed), as hierarchical-3DGS does after its cut")

    s = sub.add_parser(parents=[common], name="train", help="per-chunk gsplat training, fanned out over the work queue")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--steps", type=int, default=30000)
    s.add_argument("--only", nargs="*", help="chunk name substrings: run just these")
    s.add_argument("--preset", default="default",
                   help="gsplat trainer config: default (ADC densification) or "
                        "mcmc (relocation up to --strategy.cap-max gaussians)")
    s.add_argument("--tag", help="write to splat_<tag>/ with its own queue stage, "
                                 "so experiments sit beside the baseline run")
    s.add_argument("--scene-scale-m", type=float,
                   help="pin gsplat's scene scale to this many metres for every tile "
                        "(default: the config's train.scene_scale_m, else gsplat "
                        "derives it from camera spread, which makes densification "
                        "depend on how far a tile's cameras reach)")
    s.add_argument("extra", nargs="*", help="extra flags passed to the trainer")

    s = sub.add_parser(parents=[common], name="verify", help="eyeball a new camera/format: EAC layout, GPS, views")
    s.add_argument("video", type=Path)
    s.add_argument("--out", type=Path, default=Path("data/verify"))
    s.add_argument("--at", type=float, default=5.0, help="seconds into the clip")
    s.add_argument("--profile", help="camera profile name (default: auto-detect)")
    s.add_argument("--hwaccel", help="e.g. cuda")

    s = sub.add_parser(parents=[common], name="eval", help="compare checkpoints on visible (unmasked) pixels")
    s.add_argument("--chunk", required=True, type=Path)
    s.add_argument("ckpts", nargs="+", type=Path)
    s.add_argument("--test-every", type=int, default=8)
    s.add_argument("--names", type=Path,
                   help="score only held-out views named in this file (one image "
                        "name per line): the shared held-out set of the variants "
                        "being compared, see holdout.common_holdout")
    s.add_argument("--at-width", type=int,
                   help="render and score at this width instead of the training width. "
                        "Required to compare models trained at different resolutions: at "
                        "their own widths the higher-resolution one is scored against a "
                        "harder target and the numbers are not comparable")

    s = sub.add_parser(parents=[common], name="seams", help="do neighbouring chunks agree about the road's height?")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--within", type=Path,
                   help="another world's chunks dir: restrict to chunks inside its "
                        "footprint. Comparing a whole survey against a bake of one "
                        "street otherwise credits the small bake for not containing "
                        "the survey's sparse fringes, which is where bad seams live")
    s.add_argument("--fail-over", type=float, metavar="M",
                   help="exit non-zero if the worst seam exceeds M metres (for CI)")

    s = sub.add_parser(parents=[common], name="merge", help="chunk splats -> one streamable world (tiles + world.json)")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--keep-floaters", action="store_true",
                   help="skip corridor pruning (keeps gaussians no camera observed)")
    s.add_argument("--single", action="store_true", help="also write one world.ply")

    s = sub.add_parser(parents=[common], name="run", help="one capture end to end (leader/worker, for a JobSet)")
    s.add_argument("--capture", required=True, type=Path, help="dir holding capture.json and video/")
    s.add_argument("--out", required=True, type=Path, help="where world.json and tiles/ are published")
    s.add_argument("--role", choices=["leader", "worker"], default="leader")
    s.add_argument("--site", type=Path, help="the baked world, for levelling against its lidar")
    s.add_argument("--work", type=Path, help="scratch (default: <out>/.work)")
    s.add_argument("--seam-fail-over", default="3.0", metavar="M",
                   help="refuse to train if the worst seam between neighbouring "
                        "chunks exceeds M metres (default 3.0; 'none' disables). "
                        "Seams are measurable after poses and before training, so "
                        "this rejects an undriveable world for the cost of poses "
                        "rather than the cost of the whole bake")

    s = sub.add_parser(parents=[common], name="lod", help="cheaper copies of a merged world's tiles (far / probe LOD)")
    s.add_argument("--world", required=True, type=Path, help="merge output dir")
    s.add_argument("--keep", type=float, default=0.125,
                   help="fraction of gaussians to keep, ranked by opacity x footprint")
    s.add_argument("--sh", action="store_true",
                   help="keep spherical harmonics (default: drop bands 1-3, ~73%% of the bytes)")
    s.add_argument("--name", default="far", help="LOD name; written to tiles_<name>/")

    s = sub.add_parser(parents=[common], name="mesh", help="trained splat -> textured mesh (depth fusion)")
    s.add_argument("--chunk", required=True, type=Path)
    s.add_argument("--ckpt", type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--voxel-m", type=float, default=0.05)
    s.add_argument("--depth-max-m", type=float, default=30.0)
    s.add_argument("--max-tris", type=int, default=0, help="0 = no decimation")
    s.add_argument("--min-alpha", type=float, default=0.6,
                   help="skip pixels the splat barely covers")
    s.add_argument("--image-factor", type=int, default=2,
                   help="downscale frames before CPU fusion (1 = full res)")
    s.add_argument("--edge-rel", type=float, default=0.05,
                   help="drop depth pixels whose gradient exceeds this fraction "
                        "of depth (silhouette bleed); 0 disables")

    s = sub.add_parser(parents=[common], name="drive", help="render a drive along the capture corridor")
    s.add_argument("--chunk", required=True, type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--ckpt", type=Path)
    s.add_argument("--corridor", type=Path, help="defaults to the chunk's corridor.json")
    s.add_argument("--width", type=int, default=1280)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--fov", type=float, default=90.0)
    s.add_argument("--spacing-m", type=float, default=0.35, help="metres per frame")
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--height-offset-m", type=float, default=0.0,
                   help="raise/lower from the original lens height")
    s.add_argument("--pass", dest="pass_index", type=int,
                   help="render this corridor pass (0-based, in corridor.json "
                        "order) instead of the longest one")
    s.add_argument("--all-passes", action="store_true",
                   help="render every pass, one video each (drive_passN.mp4)")

    s = sub.add_parser(parents=[common], name="route", help="corridor -> one driveable point-to-point stage")
    s.add_argument("--world", required=True, type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--join-m", type=float, default=60.0,
                   help="max endpoint gap that still counts as connected")
    s.add_argument("--dedupe-m", type=float, default=20.0,
                   help="how close counts as retracing the same road")

    s = sub.add_parser(parents=[common], name="export", help="world corridor -> road/centerline for a sim or GIS")
    s.add_argument("--world", required=True, type=Path, help="merge output dir")
    s.add_argument("--out", type=Path)
    s.add_argument("--width-m", type=float, default=6.0, help="road ribbon width")
    s.add_argument("--drop-m", type=float, default=2.4,
                   help="camera height above the road surface")
    s.add_argument("--z-up", action="store_true", help="keep ENU Z-up (default Y-up)")
    s.add_argument("--route", type=Path, help="route.json: export one stage, not every pass")

    s = sub.add_parser(parents=[common], name="smoke", help="end-to-end sanity check on the .360 sample")
    s.add_argument("--sample", type=Path, default=Path("data/samples/GS010513.360"))
    s.add_argument("--out", type=Path, default=Path("data/smoke"))

    args = p.parse_args()

    if args.cmd == "ingest":
        from .ingest import ingest_videos
        cfg = _cfg(args.config, "ingest")
        view_cfg = {k: v for k, v in cfg.items()
                    if k in ("fov", "px_per_deg", "pitch", "aspect",
                             "view_overlap_deg", "min_coverage")}
        if args.fov:
            view_cfg["fov"] = args.fov
        if args.px_per_deg:
            view_cfg["px_per_deg"] = args.px_per_deg
        if args.pitch:
            view_cfg["pitch"] = [float(x) for x in args.pitch.split(",")]
        ingest_videos(
            args.videos, args.out,
            profile=args.profile or cfg.get("profile"),
            views=cfg.get("views"), view_cfg=view_cfg,
            extract_fps=args.extract_fps or cfg.get("extract_fps", 6.0),
            spacing_m=args.spacing_m if args.spacing_m is not None
            else cfg.get("spacing_m", 1.75),
            jpeg_quality=cfg.get("jpeg_quality", 95),
            hwaccel=args.hwaccel or cfg.get("hwaccel"),
            jobs=args.jobs or cfg.get("jobs", 0),
            start_s=args.start_s, duration_s=args.duration_s,
            # `near`/`radius_m` fall back to the config like every other ingest
            # setting. A campaign that is DEFINED by a place -- one street at high
            # resolution -- could otherwise not say so in its own file, and the
            # location would live in whatever shell history invoked it.
            near=_near(args.near if args.near else cfg.get("near")),
            radius_m=args.radius_m or cfg.get("radius_m", 400.0),
            segment_s=args.segment_s,
            no_telemetry=args.no_telemetry)

    elif args.cmd == "mapillary":
        from .mapillary import fetch
        cfg = _cfg(args.config, "ingest")
        fetch(args.bbox, args.out, views=cfg.get("views"),
              max_images=args.max_images, min_seq_len=args.min_seq_len)

    elif args.cmd == "chunk":
        from .chunks import make_chunks
        cfg = _cfg(args.config, "chunk")
        make_chunks(args.frames,
                    cell_m=args.cell_m if args.cell_m is not None
                    else cfg.get("cell_m", 200.0),
                    overlap_m=args.overlap_m if args.overlap_m is not None
                    else cfg.get("overlap_m", 40.0),
                    min_frames=args.min_frames if args.min_frames is not None
                    else cfg.get("min_frames", 20),
                    corridor_cfg=_cfg(args.config, "corridor"),
                    origin=_origin(args.origin or cfg.get("origin")))

    elif args.cmd == "profiles":
        from . import profiles as P
        from . import viewplan
        from .drivers import get_driver
        if args.video:
            prof = P.detect(args.video)
        else:
            for name, pr in sorted(P.all_profiles().items()):
                lenses = ", ".join(f"{l.name}@{l.yaw_deg:+.0f}" for l in pr.lenses)
                print(f"{name:22s} {pr.driver:14s} {pr.status:10s} "
                      f"telemetry={pr.telemetry:8s} lenses: {lenses}")
                for line in (pr.match.get("track_size") or []):
                    print(f"{'':22s}   matches {line[0]}x{line[1]}")
            return
        if args.plan:
            d = get_driver(prof)
            d.prepare_sizes(P._streams(args.video))
            print(viewplan.describe(viewplan.plan_views(d, _cfg(args.config, "ingest")), d))

    elif args.cmd == "flatten":
        from .flatten import flatten
        flatten(args.video, args.out, width=args.width, fps=args.fps,
                start_s=args.start_s, duration_s=args.duration_s,
                hwaccel=args.hwaccel, profile=args.profile)

    elif args.cmd == "mask":
        from .mask import build
        build(args.frames, sample=args.sample, search_from=args.search_from,
              dark_pct=args.dark_pct, seam_band_deg=args.seam_band_deg)

    elif args.cmd == "status":
        from .poses import list_chunks
        from .workqueue import WorkQueue
        chunks = list_chunks(args.chunks)
        stages = ["poses", "train"] if args.stage == "all" else [args.stage]
        for stage in stages:
            st = WorkQueue(args.chunks, stage).status(chunks)
            print(f"{stage:6s} done={len(st['done'])} running={len(st['running'])} "
                  f"failed={len(st['failed'])} pending={len(st['pending'])}"
                  + (f"  FAILED: {', '.join(st['failed'][:6])}" if st["failed"] else ""))

    elif args.cmd == "poses":
        from .poses import solve_all
        cfg = _cfg(args.config, "poses")
        solve_all(args.chunks,
                  matcher=args.matcher or cfg.get("matcher", "spatial"),
                  align=not args.no_align and cfg.get("align", True),
                  only=args.only,
                  spatial_radius=args.spatial_radius or cfg.get("spatial_radius", 4),
                  refresh=args.refresh, mapper=args.mapper,
                  loop_closure=args.loop_closure or cfg.get("loop_closure", "none"))

    elif args.cmd == "global-solve":
        from .globalsfm import solve_global
        pcfg = _cfg(args.config, "poses")
        ccfg = _cfg(args.config, "chunk")
        origin = _origin(args.origin or ccfg.get("origin"))
        if origin is None:
            raise SystemExit("[global] no origin: pin one in the config's chunk "
                             "section or pass --origin, or the global model will "
                             "not share a frame with the chunk bounds")
        solve_global(args.frames, origin,
                     matcher=args.matcher,
                     align=pcfg.get("align", True),
                     spatial_radius=(args.spatial_radius if args.spatial_radius
                                     is not None else pcfg.get("spatial_radius", 4)),
                     mapper=args.mapper,
                     loop_closure=(args.loop_closure or
                                   pcfg.get("loop_closure", "none")))

    elif args.cmd == "global-split":
        from .globalsfm import split_chunk
        from .poses import list_chunks
        chunks = list_chunks(args.chunks)
        if not chunks:
            raise SystemExit(f"[global] no chunks under {args.chunks}")
        from .globalsfm import relocated_frames
        rows = [split_chunk(args.frames, c, halo_m=args.halo_m,
                            global_sparse=args.sparse, rule=args.rule,
                            refine=args.refine)
                for c in chunks]
        empty = [r["chunk"] for r in rows if r["images"] < 2]
        print(f"[global] split {len(rows)} chunks; "
              f"{sum(r['images'] for r in rows)} image memberships, "
              f"{sum(r['linked'] for r in rows)} new links")
        holes = []
        for name in empty:
            chunk = args.chunks / name
            dealt, registered = relocated_frames(args.frames, chunk, args.sparse)
            if dealt and registered >= 0.9 * dealt:
                # the GPS dealt these frames to a cell the solve says they are
                # not in; the model has them, in the neighbours' cuts
                parked = args.chunks / "relocated" / name
                parked.parent.mkdir(exist_ok=True)
                if parked.exists():
                    shutil.rmtree(parked)
                shutil.move(str(chunk), str(parked))
                print(f"[global] {name}: empty because its {dealt} frames are "
                      f"registered elsewhere ({registered} in the global model): "
                      f"a GPS-dealt cell, not a hole. Parked at {parked}")
            else:
                holes.append((name, dealt, registered))
        if holes:
            raise SystemExit(f"[global] {len(holes)} chunk(s) came out empty with "
                             f"frames the global model never registered: {holes}")

    elif args.cmd == "train":
        from .train import train_all
        tcfg = _cfg(args.config, "train")
        train_all(args.chunks, steps=args.steps, extra=args.extra, only=args.only,
                  preset=args.preset, tag=args.tag,
                  scene_scale_m=(args.scene_scale_m if args.scene_scale_m is not None
                                 else tcfg.get("scene_scale_m")))

    elif args.cmd == "verify":
        from .verify import verify
        verify(args.video, args.out, at_s=args.at, profile=args.profile,
               view_cfg=_cfg(args.config, "ingest"), hwaccel=args.hwaccel)

    elif args.cmd == "eval":
        from .evaluate import evaluate
        evaluate(args.chunk, args.ckpts, test_every=args.test_every, at_width=args.at_width,
                 names=args.names)

    elif args.cmd == "run":
        from .run import run
        sfo = None if str(args.seam_fail_over).lower() in ("none", "off", "0") \
            else float(args.seam_fail_over)
        run(args.capture, args.out, args.role, args.config, site=args.site,
            work=args.work, seam_fail_over=sfo)

    elif args.cmd == "lod":
        from .lod import build
        build(args.world, keep=args.keep, sh=args.sh, name=args.name)

    elif args.cmd == "seams":
        from .seams import report
        return report(args.chunks, within=args.within, fail_over=args.fail_over)

    elif args.cmd == "merge":
        from .merge import merge
        merge(args.chunks, args.out, prune_corridor=not args.keep_floaters,
              single=args.single)

    elif args.cmd == "mesh":
        from .mesh import build
        build(args.chunk, args.ckpt, args.out, voxel_m=args.voxel_m,
              depth_max_m=args.depth_max_m, max_tris=args.max_tris,
              min_alpha=args.min_alpha, image_factor=args.image_factor,
              edge_rel=args.edge_rel)

    elif args.cmd == "drive":
        from .drive import render
        kw = dict(ckpt=args.ckpt, corridor=args.corridor, width=args.width,
                  height=args.height, fov_deg=args.fov, spacing_m=args.spacing_m,
                  fps=args.fps, height_offset_m=args.height_offset_m)
        if args.all_passes:
            import json as _json
            cor = _json.loads(Path(args.corridor or args.chunk / "corridor.json").read_text())
            out = Path(args.out or args.chunk / "drive")
            for i in range(len(cor.get("passes") or [])):
                render(args.chunk, out / f"pass{i}", pass_index=i, **kw)
        else:
            render(args.chunk, args.out, pass_index=args.pass_index, **kw)

    elif args.cmd == "route":
        from .route import build_from_world
        build_from_world(args.world, args.out, join_m=args.join_m,
                         dedupe_tol_m=args.dedupe_m)

    elif args.cmd == "export":
        from .export import export
        export(args.world, args.out, width_m=args.width_m, y_up=not args.z_up,
               drop_m=args.drop_m, route=args.route)

    elif args.cmd == "smoke":
        from .chunks import make_chunks
        from .ingest import ingest_videos
        from .poses import solve_all
        if not args.sample.exists():
            raise SystemExit(f"{args.sample} not found — run `just fetch-360` "
                             "(CC BY 4.0, see THIRD_PARTY.md)")
        # indoor sample: GPS is jitter, so keep time-spaced frames and use
        # sequential matching with no geo alignment
        frames = ingest_videos([args.sample], args.out, extract_fps=0.25, spacing_m=0)
        chunks = make_chunks(frames, cell_m=0)
        solve_all(chunks, matcher="sequential", align=False)
        print("[smoke] poses OK — run `splatpipe train --chunks "
              f"{chunks}` on a GPU box with gsplat to finish")


if __name__ == "__main__":
    # main() RETURNS a status for the subcommands that can fail (seams
    # --fail-over). Dropping it here would make the failure print and the
    # process succeed, which is worse than having no check at all.
    sys.exit(main())
