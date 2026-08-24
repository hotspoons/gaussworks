# SPDX-License-Identifier: Apache-2.0
"""splatpipe CLI: ingest -> chunk -> poses -> train."""

import argparse
from pathlib import Path

import yaml


def _cfg(path: str | None, stage: str) -> dict:
    if not path:
        return {}
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get(stage, {}) or {}


def main():
    p = argparse.ArgumentParser(prog="splatpipe")
    p.add_argument("--config", help="stage-defaults yaml (see configs/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="video(s) -> geotagged pinhole frames")
    s.add_argument("videos", nargs="+", type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--projection", choices=["equirect", "flat"])
    s.add_argument("--extract-fps", type=float)
    s.add_argument("--spacing-m", type=float)
    s.add_argument("--hwaccel", help="ffmpeg decoder, e.g. cuda (8K HEVC is decode-bound)")
    s.add_argument("--start-s", type=float, default=0.0, help="skip into the clip")
    s.add_argument("--duration-s", type=float, help="ingest only this many seconds")

    s = sub.add_parser("mapillary", help="fetch 360 sequences w/ GPS from Mapillary")
    s.add_argument("--bbox", required=True, help="w,s,e,n")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--max-images", type=int, default=2000)
    s.add_argument("--min-seq-len", type=int, default=50)

    s = sub.add_parser("chunk", help="frames -> overlapping locality chunks")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--cell-m", type=float, help="grid cell size, metres (0 = one chunk)")
    s.add_argument("--overlap-m", type=float, help="halo pulled in from neighbours")
    s.add_argument("--min-frames", type=int, help="cells with fewer own frames are dropped")

    s = sub.add_parser("mask", help="auto-mask the capture vehicle out of every frame")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--sample", type=int, default=60)
    s.add_argument("--search-from", type=float, default=0.35,
                   help="fraction down the frame where the rig may start")
    s.add_argument("--dark-pct", type=float, default=45.0,
                   help="percentile of median luminance treated as rig")

    s = sub.add_parser("status", help="queue state across chunks (pending/running/done/failed)")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--stage", default="all", choices=["all", "poses", "train"])

    s = sub.add_parser("poses", help="per-chunk COLMAP/GLOMAP + ENU alignment")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--matcher", choices=["spatial", "sequential", "exhaustive"])
    s.add_argument("--no-align", action="store_true")
    s.add_argument("--only", nargs="*", help="chunk name substrings: run just these")

    s = sub.add_parser("train", help="per-chunk gsplat training, fanned out over the work queue")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--steps", type=int, default=30000)
    s.add_argument("--only", nargs="*", help="chunk name substrings: run just these")
    s.add_argument("extra", nargs="*", help="extra flags passed to the trainer")

    s = sub.add_parser("verify", help="eyeball a new camera/format: EAC layout, GPS, views")
    s.add_argument("video", type=Path)
    s.add_argument("--out", type=Path, default=Path("data/verify"))
    s.add_argument("--at", type=float, default=5.0, help="seconds into the clip")

    s = sub.add_parser("merge", help="chunk splats -> one streamable world (tiles + world.json)")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--keep-floaters", action="store_true",
                   help="skip corridor pruning (keeps gaussians no camera observed)")
    s.add_argument("--single", action="store_true", help="also write one world.ply")

    s = sub.add_parser("mesh", help="trained splat -> textured mesh (depth fusion)")
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

    s = sub.add_parser("drive", help="render a drive along the capture corridor")
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

    s = sub.add_parser("route", help="corridor -> one driveable point-to-point stage")
    s.add_argument("--world", required=True, type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--join-m", type=float, default=60.0,
                   help="max endpoint gap that still counts as connected")
    s.add_argument("--dedupe-m", type=float, default=20.0,
                   help="how close counts as retracing the same road")

    s = sub.add_parser("export", help="world corridor -> road/centerline for a sim or GIS")
    s.add_argument("--world", required=True, type=Path, help="merge output dir")
    s.add_argument("--out", type=Path)
    s.add_argument("--width-m", type=float, default=6.0, help="road ribbon width")
    s.add_argument("--drop-m", type=float, default=2.4,
                   help="camera height above the road surface")
    s.add_argument("--z-up", action="store_true", help="keep ENU Z-up (default Y-up)")
    s.add_argument("--route", type=Path, help="route.json: export one stage, not every pass")

    s = sub.add_parser("smoke", help="end-to-end sanity check on the .360 sample")
    s.add_argument("--sample", type=Path, default=Path("data/samples/GS010513.360"))
    s.add_argument("--out", type=Path, default=Path("data/smoke"))

    args = p.parse_args()

    if args.cmd == "ingest":
        from .ingest import ingest_videos
        cfg = _cfg(args.config, "ingest")
        ingest_videos(
            args.videos, args.out,
            projection=args.projection or cfg.get("projection", "equirect"),
            views=cfg.get("views"),
            extract_fps=args.extract_fps or cfg.get("extract_fps", 6.0),
            spacing_m=args.spacing_m if args.spacing_m is not None
            else cfg.get("spacing_m", 1.75),
            jpeg_quality=cfg.get("jpeg_quality", 95),
            hwaccel=args.hwaccel or cfg.get("hwaccel"),
            start_s=args.start_s, duration_s=args.duration_s)

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
                    corridor_cfg=_cfg(args.config, "corridor"))

    elif args.cmd == "mask":
        from .mask import build
        build(args.frames, sample=args.sample, search_from=args.search_from,
              dark_pct=args.dark_pct)

    elif args.cmd == "status":
        from .poses import list_chunks
        from .queue import WorkQueue
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
                  only=args.only)

    elif args.cmd == "train":
        from .train import train_all
        train_all(args.chunks, steps=args.steps, extra=args.extra, only=args.only)

    elif args.cmd == "verify":
        from .verify import verify
        verify(args.video, args.out, at_s=args.at)

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
        render(args.chunk, args.out, ckpt=args.ckpt, corridor=args.corridor,
               width=args.width, height=args.height, fov_deg=args.fov,
               spacing_m=args.spacing_m, fps=args.fps,
               height_offset_m=args.height_offset_m)

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
    main()
