# SPDX-License-Identifier: Apache-2.0
"""Stage 1: video -> geotagged, distance-spaced pinhole frames.

- GPS comes from GoPro GPMF via exiftool (-ee); timestamps are taken relative
  to the first GPS sample, which GoPro emits at recording start, so frame time
  (idx / fps) maps onto the GPS timeline directly. Good to well under a meter
  at driving speeds; RTK-grade sync can come later if it ever matters.
- Candidate frames are extracted at extract_fps, then one frame is kept per
  spacing_m of travel: the sharpest (Laplacian variance) in each window.
- .360 (GoPro EAC) is handled natively: both video tracks are extracted and
  splatpipe.eac remaps EAC -> pinhole views in a single resample.
"""

import datetime as dt
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .eac import EacSampler, view_dirs
from .geo import track_distances
from .reframe import DEFAULT_VIEWS
from .writer import FrameWriter


def _run_json(cmd: list[str]):
    return json.loads(subprocess.run(cmd, check=True, capture_output=True).stdout)


def video_fps(video: Path) -> float:
    probe = _run_json(["ffprobe", "-v", "error", "-select_streams", "v:0",
                       "-show_entries", "stream=avg_frame_rate", "-of", "json", str(video)])
    num, den = probe["streams"][0]["avg_frame_rate"].split("/")
    return float(num) / float(den)


def _parse_gps_time(ts: str) -> float:
    ts = ts.replace("Z", "").strip()
    fmt = "%Y:%m:%d %H:%M:%S.%f" if "." in ts else "%Y:%m:%d %H:%M:%S"
    return dt.datetime.strptime(ts, fmt).timestamp()


def extract_telemetry(video: Path) -> list[dict]:
    """GPMF GPS track as [{t, lat, lon, alt}], t relative to the first sample."""
    rec = _run_json(["exiftool", "-ee", "-n", "-j", "-G3",
                     "-api", "LargeFileSupport=1", str(video)])[0]
    docs: dict[int, dict] = {}
    for key, val in rec.items():
        if ":" not in key:
            continue
        group, tag = key.split(":", 1)
        if group.startswith("Doc"):
            docs.setdefault(int(group[3:]), {})[tag] = val
    pts = []
    for i in sorted(docs):
        d = docs[i]
        if "GPSLatitude" in d and "GPSLongitude" in d:
            pts.append(d)
    if not pts:
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


def extract_candidates_360(video: Path, tmp: Path, extract_fps: float,
                           hwaccel: str | None = None, start_s: float = 0.0,
                           duration_s: float | None = None
                           ) -> list[tuple[float, Path, Path]]:
    """GoPro .360: dump matching frame pairs from both EAC video tracks."""
    kw = dict(hwaccel=hwaccel, start_s=start_s, duration_s=duration_s)
    t1 = extract_candidates(video, tmp / "t1", extract_fps, stream="0:v:0", **kw)
    t2 = extract_candidates(video, tmp / "t2", extract_fps, stream="0:v:1", **kw)
    if len(t1) != len(t2):
        print(f"[ingest] warning: track frame counts differ ({len(t1)} vs {len(t2)})")
    return [(t, p1, p2) for (t, p1), (_, p2) in zip(t1, t2)]


def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    if img.shape[1] > 1024:  # sharpness ranking doesn't need full res
        scale = 1024 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def ingest_videos(videos: list[Path], out: Path, projection: str = "equirect",
                  views: list[dict] | None = None, extract_fps: float = 6.0,
                  spacing_m: float = 1.75, jpeg_quality: int = 95,
                  hwaccel: str | None = None, start_s: float = 0.0,
                  duration_s: float | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    is_360 = any(v.suffix.lower() == ".360" for v in videos)
    writer = FrameWriter(out, projection="eac" if is_360 else projection,
                         views=views, jpeg_quality=jpeg_quality)
    eac_sampler = None
    view_specs = views or DEFAULT_VIEWS
    try:
        for video in videos:
            native_360 = video.suffix.lower() == ".360"
            gps = extract_telemetry(video)
            with tempfile.TemporaryDirectory(dir=out, prefix=".candidates-") as td:
                if native_360:
                    pairs = extract_candidates_360(video, Path(td), extract_fps,
                                                   hwaccel=hwaccel, start_s=start_s,
                                                   duration_s=duration_s)
                    cands = [(t, p1) for t, p1, _ in pairs]
                else:
                    cands = extract_candidates(video, Path(td), extract_fps,
                                               hwaccel=hwaccel, start_s=start_s,
                                               duration_s=duration_s)
                if gps:
                    gt = np.array([p["t"] for p in gps])
                    lat = np.interp([t for t, _ in cands], gt, [p["lat"] for p in gps])
                    lon = np.interp([t for t, _ in cands], gt, [p["lon"] for p in gps])
                    alt = np.interp([t for t, _ in cands], gt, [p["alt"] for p in gps])
                    dist = track_distances(list(lat), list(lon))
                else:
                    lat = lon = alt = [None] * len(cands)
                    dist = None

                if dist is None or spacing_m <= 0:
                    keep = range(len(cands))  # no GPS / spacing off: keep all
                else:
                    keep, window, next_d = [], [], 0.0
                    for i in range(len(cands)):
                        window.append(i)
                        if dist[i] >= next_d:
                            keep.append(max(window, key=lambda j: sharpness(cands[j][1])))
                            window, next_d = [], dist[i] + spacing_m
                for i in keep:
                    t, path = cands[i]
                    img = cv2.imread(str(path))
                    if img is None:
                        continue
                    if native_360:
                        img2 = cv2.imread(str(pairs[i][2]))
                        if img2 is None:
                            continue
                        if eac_sampler is None:
                            eac_sampler = EacSampler(img.shape[1], img.shape[0])
                            eac_dirs = [view_dirs(v["width"], v["height"], v["fov"],
                                                  v["yaw"], v["pitch"]) for v in view_specs]
                        cams = [eac_sampler.sample(("view", k), d, img, img2)
                                for k, d in enumerate(eac_dirs)]
                        writer.add_views(cams, lat[i], lon[i], alt[i], t=t,
                                         meta={"video": video.name})
                    else:
                        writer.add(img, lat[i], lon[i], alt[i], t=t, meta={"video": video.name})
            print(f"[ingest] {video.name}: gps_samples={len(gps)} kept={writer.seq}")
    finally:
        writer.close()
    return out


def ingest_images(images: list[Path], out: Path) -> Path:
    """Flat, pre-framed images (toy datasets, borrowed front cam) -> layout."""
    out.mkdir(parents=True, exist_ok=True)
    writer = FrameWriter(out, projection="flat")
    try:
        for path in images:
            img = cv2.imread(str(path))
            if img is not None:
                writer.add(img, None, None, None, meta={"source": path.name})
    finally:
        writer.close()
    print(f"[ingest] {writer.seq} images -> {out}")
    return out
