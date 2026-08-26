# SPDX-License-Identifier: Apache-2.0
"""Stage 1: video -> geotagged, distance-spaced pinhole frames.

Nothing in here knows what camera shot the footage. A profile
(splatpipe/profiles.py) names a driver (splatpipe/drivers/) that turns ray
directions into pixels, and a view plan (splatpipe/viewplan.py) decides which
rays to ask for. Supporting a new 360 camera is a YAML file; supporting a new
projection is one driver class. This stage only does the parts that are the
same for every camera:

- GPS comes from the video's telemetry via exiftool (-ee); timestamps are
  taken relative to the first GPS sample, which cameras emit at recording
  start, so frame time (idx / fps) maps onto the GPS timeline directly. Good
  to well under a meter at driving speeds.
- Candidate frames are extracted at extract_fps, then one frame is kept per
  spacing_m of travel: the sharpest (Laplacian variance) in each window.
- Extraction runs in SEGMENTS. Candidates are full-resolution stills (a .360
  is two 5952x1920 tracks), so dumping a whole clip at source frame rate costs
  tens of GB per chapter before a single frame is selected. Segmenting bounds
  that to a minute of footage at a time, while distances come from the GPS
  track rather than the candidate list so selection is unaffected by where the
  segment boundaries fall.
- Each planned view is rendered THROUGH ONE LENS. Where that lens has no data
  the pixels are black and the coverage mask says so; nothing is ever
  cross-faded from a second lens into a training image.
"""

import datetime as dt
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import profiles, viewplan
from .drivers import get_driver
from .eac import view_dirs
from .geo import ll_to_enu, track_distances
from .writer import FrameWriter


def _run_json(cmd: list[str]):
    return json.loads(subprocess.run(cmd, check=True, capture_output=True).stdout)


def video_duration(video: Path) -> float:
    probe = _run_json(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "json", str(video)])
    return float(probe["format"]["duration"])


def video_fps(video: Path) -> float:
    probe = _run_json(["ffprobe", "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=avg_frame_rate", "-of", "json", str(video)])
    num, den = probe["streams"][0]["avg_frame_rate"].split("/")
    return float(num) / float(den)


def _parse_gps_time(ts: str) -> float:
    ts = ts.replace("Z", "").strip()
    fmt = "%Y:%m:%d %H:%M:%S.%f" if "." in ts else "%Y:%m:%d %H:%M:%S"
    return dt.datetime.strptime(ts, fmt).timestamp()


EXIFTOOL = os.environ.get("EXIFTOOL", "exiftool")


def exiftool_version() -> float:
    try:
        out = subprocess.run([EXIFTOOL, "-ver"], check=True,
                             capture_output=True).stdout.decode().strip()
        return float(out)
    except Exception:
        return 0.0


def _doc_key(group: str):
    """'Doc12' or 'Doc1-7' -> a sortable tuple. GPS9 payloads nest."""
    return tuple(int(x) for x in group[3:].split("-") if x.isdigit())


def extract_telemetry(video: Path) -> list[dict]:
    """GPS track as [{t, lat, lon, alt}], t relative to the first sample.

    exiftool reads GoPro GPMF and Insta360's boxes through the same -ee pass,
    so the profile's `telemetry` field only decides whether to look at all.

    EXIFTOOL VERSION MATTERS. GoPro moved from the GPS5 payload to GPS9 with
    the HERO11 generation, and the MAX 2 writes GPS9. An exiftool without GPS9
    support -- including the 12.76 that Ubuntu 24.04 ships -- parses the file
    happily, reports every other GPMF stream, and returns NO GPS AT ALL. That
    failure is silent and downstream it looks like a camera with GPS switched
    off: no geo alignment, no locality chunking, and distance-based frame
    spacing quietly degrades to "keep everything". Set $EXIFTOOL or install
    12.90+; `verify` checks and says so.
    """
    rec = _run_json([EXIFTOOL, "-ee", "-n", "-j", "-G3",
                     "-api", "LargeFileSupport=1", str(video)])[0]
    docs: dict[tuple, dict] = {}
    for key, val in rec.items():
        if ":" not in key:
            continue
        group, tag = key.split(":", 1)
        if group.startswith("Doc"):
            docs.setdefault(_doc_key(group), {})[tag] = val
    pts = []
    for i in sorted(docs):
        d = docs[i]
        if "GPSLatitude" in d and "GPSLongitude" in d:
            pts.append(d)
    if not pts:
        ver = exiftool_version()
        if ver and ver < 12.90:
            print(f"[ingest] no GPS found, and {EXIFTOOL} is {ver:g} -- too old "
                  f"for GoPro GPS9 (HERO11+/MAX 2). This is almost certainly the "
                  f"reason, not the camera. Install 12.90+ or set $EXIFTOOL.")
        return []
    timed = [p for p in pts if "GPSDateTime" in p]
    if timed:
        t0 = _parse_gps_time(timed[0]["GPSDateTime"])
        out, last_t = [], -1.0
        for p in pts:
            t = _parse_gps_time(p["GPSDateTime"]) - t0 if "GPSDateTime" in p else last_t
            if t <= last_t:  # multiple samples per payload share a stamp; nudge
                t = last_t + 1e-3
            last_t = t
            out.append({"t": t, "lat": float(p["GPSLatitude"]),
                        "lon": float(p["GPSLongitude"]),
                        "alt": float(p.get("GPSAltitude", 0.0))})
        return out
    # no per-sample time: assume uniform rate across the samples
    return [{"t": float(i), "lat": float(p["GPSLatitude"]),
             "lon": float(p["GPSLongitude"]), "alt": float(p.get("GPSAltitude", 0.0))}
            for i, p in enumerate(pts)]


def extract_candidates(video: Path, tmp: Path, extract_fps: float,
                       stream: str | None = None,
                       hwaccel: str | None = None, start_s: float = 0.0,
                       duration_s: float | None = None) -> list[tuple[float, Path]]:
    """Dump candidate frames; returns [(t_seconds, path)] in order.

    8K HEVC is slow to decode on CPU and a .360 holds TWO such tracks, so a
    long capture is decode-bound. `hwaccel="cuda"` moves decode to NVDEC and
    lets the filter chain stay on the CPU, which is the cheap 5-10x.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if start_s:
        cmd += ["-ss", str(start_s)]          # before -i: seeks, does not decode
    cmd += ["-i", str(video)]
    if duration_s:
        cmd += ["-t", str(duration_s)]
    if stream is not None:
        cmd += ["-map", stream]
    cmd += ["-vf", f"fps={extract_fps}", "-qscale:v", "2", str(tmp / "%06d.jpg")]
    subprocess.run(cmd, check=True)
    paths = sorted(tmp.glob("*.jpg"))
    # timestamps stay on the source clock so GPS interpolation still lines up
    return [(start_s + (i + 0.5) / extract_fps, p) for i, p in enumerate(paths)]


def extract_candidates_multi(video: Path, tmp: Path, extract_fps: float,
                             maps: list[str | None], hwaccel: str | None = None,
                             start_s: float = 0.0, duration_s: float | None = None
                             ) -> list[tuple[float, list[Path]]]:
    """Dump matching frames from every stream the driver asked for."""
    kw = dict(hwaccel=hwaccel, start_s=start_s, duration_s=duration_s)
    per_stream = [extract_candidates(video, tmp / f"s{i}", extract_fps, stream=m, **kw)
                  for i, m in enumerate(maps)]
    n = min(len(s) for s in per_stream)
    if any(len(s) != n for s in per_stream):
        print(f"[ingest] warning: stream frame counts differ "
              f"({[len(s) for s in per_stream]}), truncating to {n}")
    return [(per_stream[0][i][0], [s[i][1] for s in per_stream]) for i in range(n)]


def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    if img.shape[1] > 1024:  # sharpness ranking doesn't need full res
        scale = 1024 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _resolve(video: Path, hint: str | None, view_cfg: dict):
    prof = profiles.detect(video, hint)
    driver = get_driver(prof)
    driver.prepare_sizes(profiles._streams(video))
    plan = viewplan.plan_views(driver, view_cfg) if driver.reframes else []
    if plan:
        print(viewplan.describe(plan, driver))
    return prof, driver, plan


def ingest_videos(videos: list[Path], out: Path, views: list[dict] | None = None,
                  view_cfg: dict | None = None, profile: str | None = None,
                  extract_fps: float = 6.0,
                  spacing_m: float = 1.75, jpeg_quality: int = 95,
                  hwaccel: str | None = None, start_s: float = 0.0,
                  duration_s: float | None = None,
                  near: tuple[float, float] | None = None,
                  radius_m: float = 400.0, segment_s: float = 60.0,
                  no_telemetry: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    view_cfg = dict(view_cfg or {})
    if views:
        view_cfg["views"] = views

    prof, driver, plan = _resolve(videos[0], profile, view_cfg)
    dirs = [view_dirs(v["width"], v["height"], v["fov"], v["yaw"], v["pitch"])
            for v in plan]
    coverage = [driver.coverage(("view", k), d, plan[k]["lens"])
                for k, d in enumerate(dirs)]
    for k, v in enumerate(plan):
        frac = float(np.mean(coverage[k]))
        if frac < 0.999:
            print(f"[ingest] cam{k} ({v['lens_name']} yaw={v['yaw']:g}) "
                  f"covers {frac:.1%} of its frame; the rest is masked")
    writer = FrameWriter(out, plan=plan, jpeg_quality=jpeg_quality,
                         profile=prof.name, coverage=coverage if plan else None)

    try:
        for video in videos:
            if video is not videos[0]:
                other, _, other_plan = _resolve(video, profile, view_cfg)
                if other.name != prof.name or other_plan != plan:
                    raise SystemExit(
                        f"[ingest] {video.name} resolves to profile "
                        f"{other.name!r} but {videos[0].name} resolved to "
                        f"{prof.name!r}. Ingest one camera at a time and merge "
                        f"at the chunk stage, where mixed rigs belong.")
            want_gps = prof.telemetry != "none" and not no_telemetry
            gps = extract_telemetry(video) if want_gps else []
            if not gps and want_gps:
                # Silently continuing here is how a run wastes hours: with no
                # track there is no distance, so spacing_m never engages and
                # --near cannot filter, and you get every extracted frame of
                # the whole clip instead of the stretch you asked for.
                detail = []
                if spacing_m > 0:
                    detail.append(f"--spacing-m {spacing_m} would keep EVERY "
                                  f"frame ({extract_fps} fps)")
                if near:
                    detail.append("--near/--radius-m cannot filter anything")
                raise SystemExit(
                    f"[ingest] {video.name}: profile {prof.name!r} expects "
                    f"{prof.telemetry} telemetry and none was found"
                    + (" -- " + "; ".join(detail) if detail else "")
                    + f".\n  Check with: splatpipe verify {video}\n"
                    f"  To ingest anyway, use --spacing-m 0 --no-telemetry.")
            if gps:
                gt = np.array([p["t"] for p in gps])
                glat = np.array([p["lat"] for p in gps])
                glon = np.array([p["lon"] for p in gps])
                galt = np.array([p["alt"] for p in gps])
                # distance along the GPS track, so a candidate's distance
                # depends only on its timestamp -- segment boundaries cannot
                # shift the selection
                gdist = np.array(track_distances(list(glat), list(glon)))
            centre = (near[0], near[1], 0.0) if near else None
            if centre is not None and gps:
                # distance of every GPS sample from the point of interest, so
                # whole segments the car never came near can be skipped
                # without decoding them. A --near filter over three 8K
                # chapters is otherwise decode-bound on footage it will throw
                # away: the filter used to run AFTER extraction.
                gnear = np.array([math.hypot(*ll_to_enu(la, lo, 0.0, centre)[:2])
                                  for la, lo in zip(glat, glon)])
            maps = driver.ffmpeg_maps()

            end_s = start_s + duration_s if duration_s else video_duration(video)
            next_d, kept_here, seen_here = 0.0, 0, 0
            seg_start = start_s
            skipped_s = 0.0
            while seg_start < end_s - 1e-3:
                seg_len = min(segment_s, end_s - seg_start)
                if centre is not None and gps:
                    # GPS is ~10 Hz and interpolated per candidate, so a
                    # segment whose nearest sample (with a little slack for
                    # the interpolation and the segment edges) is beyond the
                    # radius cannot contribute a frame. Skip the decode.
                    sel = (gt >= seg_start - 1.0) & (gt <= seg_start + seg_len + 1.0)
                    if not sel.any() or gnear[sel].min() > radius_m + 25.0:
                        seg_start += seg_len
                        skipped_s += seg_len
                        continue
                with tempfile.TemporaryDirectory(dir=out, prefix=".cand-") as td:
                    cands = extract_candidates_multi(video, Path(td), extract_fps,
                                                     maps, hwaccel=hwaccel,
                                                     start_s=seg_start,
                                                     duration_s=seg_len)
                    seen_here += len(cands)
                    if not cands:
                        seg_start += seg_len
                        continue

                    times = np.array([t for t, _ in cands])
                    if gps:
                        lat = np.interp(times, gt, glat)
                        lon = np.interp(times, gt, glon)
                        alt = np.interp(times, gt, galt)
                        dist = np.interp(times, gt, gdist)
                    else:
                        lat = lon = alt = [None] * len(cands)
                        dist = None

                    if dist is None or spacing_m <= 0:
                        keep = list(range(len(cands)))
                    else:
                        keep, window = [], []
                        for i in range(len(cands)):
                            window.append(i)
                            if dist[i] >= next_d:
                                keep.append(max(window,
                                                key=lambda j: sharpness(cands[j][1][0])))
                                window, next_d = [], dist[i] + spacing_m

                    if centre is not None and gps:
                        keep = [i for i in keep
                                if sum(c * c for c in
                                       ll_to_enu(lat[i], lon[i], 0.0, centre)[:2])
                                <= radius_m * radius_m]

                    for i in keep:
                        t, paths = cands[i]
                        frames = [cv2.imread(str(p)) for p in paths]
                        if any(f is None for f in frames):
                            continue
                        if plan:
                            cams = [driver.sample(("view", k), d, frames,
                                                  plan[k]["lens"])
                                    for k, d in enumerate(dirs)]
                        else:
                            cams = [frames[0]]
                        writer.add_views(cams, lat[i], lon[i], alt[i], t=t,
                                         meta={"video": video.name})
                        kept_here += 1
                seg_start += seg_len
                print(f"[ingest] {video.name}: {seg_start - start_s:6.0f}s / "
                      f"{end_s - start_s:.0f}s, kept {kept_here}", flush=True)
            print(f"[ingest] {video.name}: gps_samples={len(gps)} "
                  f"candidates={seen_here} kept={kept_here}"
                  + (f" (skipped {skipped_s:.0f}s of footage outside "
                     f"--near/--radius-m without decoding)" if skipped_s else ""),
                  flush=True)
    finally:
        writer.close()
    return out


def ingest_images(images: list[Path], out: Path) -> Path:
    """Flat, pre-framed images (toy datasets, borrowed front cam) -> layout."""
    out.mkdir(parents=True, exist_ok=True)
    writer = FrameWriter(out)
    try:
        for path in images:
            img = cv2.imread(str(path))
            if img is not None:
                writer.add(img, None, None, None, meta={"source": path.name})
    finally:
        writer.close()
    print(f"[ingest] {writer.seq} images -> {out}")
    return out
