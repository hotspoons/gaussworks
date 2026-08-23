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

    s = sub.add_parser("mapillary", help="fetch 360 sequences w/ GPS from Mapillary")
    s.add_argument("--bbox", required=True, help="w,s,e,n")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--max-images", type=int, default=2000)
    s.add_argument("--min-seq-len", type=int, default=50)

    s = sub.add_parser("chunk", help="frames -> overlapping spatial chunks")
    s.add_argument("--frames", required=True, type=Path)
    s.add_argument("--chunk-m", type=float)
    s.add_argument("--overlap-m", type=float)

    s = sub.add_parser("poses", help="per-chunk COLMAP/GLOMAP + ENU alignment")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--matcher", choices=["spatial", "sequential", "exhaustive"])
    s.add_argument("--no-align", action="store_true")

    s = sub.add_parser("train", help="per-chunk gsplat training, rank-sharded")
    s.add_argument("--chunks", required=True, type=Path)
    s.add_argument("--steps", type=int, default=30000)
    s.add_argument("extra", nargs="*", help="extra flags passed to the trainer")

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
            jpeg_quality=cfg.get("jpeg_quality", 95))

    elif args.cmd == "mapillary":
        from .mapillary import fetch
        cfg = _cfg(args.config, "ingest")
        fetch(args.bbox, args.out, views=cfg.get("views"),
              max_images=args.max_images, min_seq_len=args.min_seq_len)

    elif args.cmd == "chunk":
        from .chunks import make_chunks
        cfg = _cfg(args.config, "chunk")
        make_chunks(args.frames,
                    chunk_m=args.chunk_m if args.chunk_m is not None
                    else cfg.get("chunk_m", 200.0),
                    overlap_m=args.overlap_m if args.overlap_m is not None
                    else cfg.get("overlap_m", 40.0),
                    min_frames=cfg.get("min_frames", 20))

    elif args.cmd == "poses":
        from .poses import solve_all
        cfg = _cfg(args.config, "poses")
        solve_all(args.chunks,
                  matcher=args.matcher or cfg.get("matcher", "spatial"),
                  align=not args.no_align and cfg.get("align", True))

    elif args.cmd == "train":
        from .train import train_all
        train_all(args.chunks, steps=args.steps, extra=args.extra)

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
        chunks = make_chunks(frames, chunk_m=0)
        solve_all(chunks, matcher="sequential", align=False)
        print("[smoke] poses OK — run `splatpipe train --chunks "
              f"{chunks}` on a GPU box with gsplat to finish")


if __name__ == "__main__":
    main()
