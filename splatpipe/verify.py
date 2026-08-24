# SPDX-License-Identifier: Apache-2.0
"""Sanity-check a new camera or file format before committing a whole capture.

Run this on a 30-second driveway clip from any new camera. It reports the
container layout, dumps one decoded frame per output view, and prints the GPS
it found -- so a wrong EAC template, a missing telemetry track, or a bad mount
angle is caught in a minute instead of after a three-hour drive.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2

from .eac import EacSampler, Template, eac_to_equirect, view_dirs
from .ingest import extract_candidates, extract_candidates_360, extract_telemetry
from .reframe import DEFAULT_VIEWS, Reframer


def _streams(video: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,width,height", "-of", "json", str(video)],
        check=True, capture_output=True).stdout
    return json.loads(out)["streams"]


def _budget(video: Path):
    """Measured bitrate -> minutes of capture per card. Guessing this from spec
    sheets is how people run out of card halfway down a road."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)], check=True, capture_output=True).stdout
    try:
        secs = float(out.decode().strip())
    except ValueError:
        return
    if secs <= 0:
        return
    size = video.stat().st_size
    mbps = size * 8 / secs / 1e6
    gb_min = size / secs * 60 / 1e9
    print(f"[verify] rate: {mbps:.0f} Mbps = {gb_min:.2f} GB/min")
    for card in (64, 128, 512, 1024):
        print(f"[verify]   {card:>4d} GB card -> {card / gb_min:.0f} min "
              f"({card / gb_min * 25 / 60:.1f} miles at 25 mph)")


def verify(video: Path, out: Path, at_s: float = 5.0) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    native_360 = video.suffix.lower() == ".360"

    print(f"[verify] {video.name}  ({video.stat().st_size / 1e9:.2f} GB)")
    for st in _streams(video):
        dims = f" {st.get('width')}x{st.get('height')}" if st.get("width") else ""
        print(f"[verify]   stream {st['index']}: {st['codec_type']} "
              f"{st.get('codec_name')}{dims}")

    _budget(video)

    gps = extract_telemetry(video)
    if gps:
        print(f"[verify] GPS: {len(gps)} samples, first "
              f"{gps[0]['lat']:.6f},{gps[0]['lon']:.6f} alt {gps[0]['alt']:.1f}m, "
              f"track {gps[-1]['t'] - gps[0]['t']:.1f}s")
    else:
        print("[verify] GPS: NONE FOUND -- enable GPS on the camera, or chunking "
              "and geo-alignment will not work")

    fps = max(1.0 / max(at_s, 0.5), 0.05)   # one frame at ~at_s in
    with tempfile.TemporaryDirectory() as td:
        if native_360:
            pairs = extract_candidates_360(video, Path(td), fps)
            if not pairs:
                raise SystemExit("[verify] no frames decoded")
            t1 = cv2.imread(str(pairs[0][1]))
            t2 = cv2.imread(str(pairs[0][2]))
            tpl = Template.for_size(t1.shape[1], t1.shape[0])
            print(f"[verify] EAC tracks {t1.shape[1]}x{t1.shape[0]}, template "
                  f"side={tpl.side} face={tpl.height} blend={tpl.blend}")
            cv2.imwrite(str(out / "equirect.jpg"),
                        eac_to_equirect(t1, t2, width=min(4096, tpl.width)))
            sampler = EacSampler(t1.shape[1], t1.shape[0])
            for k, v in enumerate(DEFAULT_VIEWS):
                cams = sampler.sample(("v", k), view_dirs(
                    v["width"], v["height"], v["fov"], v["yaw"], v["pitch"]), t1, t2)
                cv2.imwrite(str(out / f"cam{k}_yaw{v['yaw']}.jpg"), cams)
        else:
            cands = extract_candidates(video, Path(td), fps)
            if not cands:
                raise SystemExit("[verify] no frames decoded")
            img = cv2.imread(str(cands[0][1]))
            h, w = img.shape[:2]
            print(f"[verify] frame {w}x{h}, aspect {w / h:.2f} "
                  f"({'equirect-like' if abs(w / h - 2) < 0.05 else 'flat/other'})")
            cv2.imwrite(str(out / "frame.jpg"), img)
            if abs(w / h - 2) < 0.05:
                for k, cam in enumerate(Reframer()(img)):
                    cv2.imwrite(str(out / f"cam{k}.jpg"), cam)

    print(f"[verify] wrote previews -> {out}")
    print("[verify] LOOK AT THEM: horizon level and straight, no seams through "
          "the middle of the road, sky up, vehicle at the bottom.")
    return out
