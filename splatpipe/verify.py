# SPDX-License-Identifier: Apache-2.0
"""Sanity-check a camera profile against real footage before a whole capture.

Run this on a 30-second driveway clip from any new camera. It resolves a
profile, reports the container layout, dumps the planned views, and prints the
GPS it found -- so a wrong template, a missing telemetry track, or a bad mount
angle is caught in a minute instead of after a three-hour drive.

It also writes the per-lens hemispheres side by side. That pair is the fastest
way to check a profile's lens geometry: the two images should agree on
everything far away and disagree only on close things, by the few centimetres
of baseline between the lenses. If they disagree about the horizon or the
roofline, the lens axes or the projection model in the profile are wrong.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import profiles, viewplan
from .drivers import get_driver
from .eac import equirect_dirs, view_dirs
from .ingest import extract_candidates_multi, extract_telemetry


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


def verify(video: Path, out: Path, at_s: float = 5.0,
           profile: str | None = None, view_cfg: dict | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)

    print(f"[verify] {video.name}  ({video.stat().st_size / 1e9:.2f} GB)")
    for st in _streams(video):
        dims = f" {st.get('width')}x{st.get('height')}" if st.get("width") else ""
        print(f"[verify]   stream {st['index']}: {st['codec_type']} "
              f"{st.get('codec_name')}{dims}")
    _budget(video)

    prof = profiles.detect(video, profile)
    driver = get_driver(prof)
    driver.prepare_sizes(profiles._streams(video))
    print(f"[verify] profile {prof.name} ({prof.status}) from {prof.source}")
    for i, lens in enumerate(prof.lenses):
        print(f"[verify]   lens {i} {lens.name:>6}: axis yaw {lens.yaw_deg:+.0f} deg, "
              f"sees {driver.half_fov_deg(i):.2f} deg from it "
              f"(using {driver.usable_half_fov_deg(i):.2f})")
    if prof.status != "validated":
        print(f"[verify] NOTE: profile status is {prof.status!r} -- the previews "
              f"below are what tells you whether its geometry is right.")

    gps = extract_telemetry(video) if prof.telemetry != "none" else []
    if gps:
        print(f"[verify] GPS: {len(gps)} samples, first "
              f"{gps[0]['lat']:.6f},{gps[0]['lon']:.6f} alt {gps[0]['alt']:.1f}m, "
              f"track {gps[-1]['t'] - gps[0]['t']:.1f}s")
    else:
        print("[verify] GPS: NONE FOUND -- enable GPS on the camera, or chunking "
              "and geo-alignment will not work")

    fps = max(1.0 / max(at_s, 0.5), 0.05)   # one frame at ~at_s in
    with tempfile.TemporaryDirectory() as td:
        cands = extract_candidates_multi(video, Path(td), fps, driver.ffmpeg_maps())
        if not cands:
            raise SystemExit("[verify] no frames decoded")
        frames = [cv2.imread(str(p)) for p in cands[0][1]]
        if any(f is None for f in frames):
            raise SystemExit("[verify] frame decode failed")

    if not driver.reframes:
        cv2.imwrite(str(out / "frame.jpg"), frames[0])
        print(f"[verify] wrote {out / 'frame.jpg'}")
        return out

    eq = equirect_dirs(2048)
    cv2.imwrite(str(out / "sphere_blended.jpg"),
                driver.sample_sphere(("eq",), eq, frames))
    for i, lens in enumerate(prof.lenses):
        img = driver.sample(("eq",), eq, frames, i)
        cv2.imwrite(str(out / f"sphere_lens{i}_{lens.name}.jpg"), img)
    if len(prof.lenses) == 2:
        # Absolute difference where BOTH lenses see the same direction. Far
        # geometry cancels; whatever is left is parallax (and exposure), which
        # is exactly what a stitcher has to invent its way around.
        a = driver.sample(("eq",), eq, frames, 0).astype(np.int16)
        b = driver.sample(("eq",), eq, frames, 1).astype(np.int16)
        both = driver.coverage(("eq",), eq, 0) & driver.coverage(("eq",), eq, 1)
        diff = np.where(both[..., None], np.abs(a - b), 0).astype(np.uint8)
        cv2.imwrite(str(out / "lens_overlap_diff.jpg"), diff)
        print(f"[verify] lenses overlap on {both.mean():.2%} of the sphere; "
              f"mean |difference| there = "
              f"{float(np.abs(a - b)[both].mean()):.1f}/255")

    plan = viewplan.plan_views(driver, view_cfg or {})
    print(viewplan.describe(plan))
    for k, v in enumerate(plan):
        d = view_dirs(v["width"], v["height"], v["fov"], v["yaw"], v["pitch"])
        cov = float(driver.coverage(("view", k), d, v["lens"]).mean())
        cv2.imwrite(str(out / f"cam{k}_{v['lens_name']}_yaw{v['yaw']:g}.jpg"),
                    driver.sample(("view", k), d, frames, v["lens"]))
        if cov < 0.999:
            print(f"[verify]   cam{k}: {cov:.1%} covered by {v['lens_name']}")

    print(f"[verify] wrote previews -> {out}")
    print("[verify] LOOK AT THEM: horizon level and straight, sky up, vehicle at "
          "the bottom, and no seam through any cam*.jpg -- each one is a single "
          "lens, so a visible boundary inside one means the profile is wrong.")
    return out
