# SPDX-License-Identifier: Apache-2.0
"""Turn a GoPro .360 into an equirectangular video any 360 player can open.

No open-source player reads .360: it is two video tracks in GoPro's own
equi-angular cubemap layout, so VLC and mpv show the raw strips rather than a
sphere. Since splatpipe already decodes that layout for the pipeline (eac.py,
validated against real Max and Max 2 footage), flattening to equirect is
nearly free -- and equirect is what every 360 player, headset and web viewer
expects.

Frames are piped straight into ffmpeg as raw video, so nothing hits disk
between the remap and the encoder.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import profiles
from .drivers import get_driver
from .eac import equirect_dirs
from .ingest import extract_candidates_multi, video_fps


def flatten(video: Path, out: Path | None = None, width: int = 4096,
            fps: float | None = None, start_s: float = 0.0,
            duration_s: float | None = None, hwaccel: str | None = None,
            crf: int = 18, profile: str | None = None) -> Path:
    video = Path(video)
    out = Path(out or video.with_suffix(".equirect.mp4"))
    rate = fps or video_fps(video)
    height = width // 2

    prof = profiles.detect(video, profile)
    driver = get_driver(prof)
    driver.prepare_sizes(profiles._streams(video))

    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        pairs = extract_candidates_multi(video, Path(td), rate,
                                         driver.ffmpeg_maps(), hwaccel=hwaccel,
                                         start_s=start_s, duration_s=duration_s)
        if not pairs:
            raise SystemExit(f"{video}: no frames decoded")
        dirs = equirect_dirs(width)
        print(f"[flatten] {len(pairs)} frames, {prof.name}"
              f" -> equirect {width}x{height} @ {rate:.2f} fps", flush=True)
        # This is the VIEWING path, so blending the lens overlap is the right
        # call here and only here -- see splatpipe/eac.py on why the same
        # blend is wrong for anything that feeds reconstruction.

        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{rate}", "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p",
             # hints for players that read them; VLC and headsets mostly rely
             # on the injected spherical box, see the note printed below
             "-metadata:s:v:0", "projection=equirectangular",
             str(out)], stdin=subprocess.PIPE)
        for i, (_, paths) in enumerate(pairs):
            frames = [cv2.imread(str(p)) for p in paths]
            if any(f is None for f in frames):
                continue
            frame = driver.sample_sphere(("equirect", width), dirs, frames)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if (i + 1) % 200 == 0:
                print(f"[flatten]   {i + 1}/{len(pairs)}", flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed")

    print(f"[flatten] wrote {out}")
    print("[flatten] play with VLC (Video > 360). If it opens flat, inject the "
          "spherical metadata first: pipx run spatialmedia -i --stereo=none "
          f"{out} {out.with_suffix('.360meta.mp4')}")
    return out
