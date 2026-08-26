# SPDX-License-Identifier: Apache-2.0
"""Stage 8: drive the capture corridor and render it.

The corridor records exactly where the camera was, so replaying it is the one
path guaranteed to stay inside observed space -- no leash logic, no flying into
the void, and the least extrapolation the reconstruction will ever be asked
for. That makes it both the best-looking output we can produce and the honest
one: if a stretch looks bad here, it is bad.

It is also the game's replay camera, and a preview of what a corridor-clamped
interactive viewer will feel like.
"""

import json
import math
import subprocess
from pathlib import Path

import numpy as np


def _resample(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Even spacing along a polyline, so speed is constant regardless of how
    the capture was decimated."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    if total <= 0:
        return points
    want = np.arange(0.0, total, spacing_m)
    return np.stack([np.interp(want, dist, points[:, k]) for k in range(3)], axis=1)


def _smooth(points: np.ndarray, window: int) -> np.ndarray:
    """Moving average: GPS jitter becomes camera shake at driving speed."""
    if window < 3 or len(points) < window:
        return points
    k = np.ones(window) / window
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], k, mode="valid")[:len(points)]
                     for c in range(3)], axis=1)


def trim_reversals(points: np.ndarray, head_m: float = 25.0,
                   turn_deg: float = 120.0) -> tuple[np.ndarray, float]:
    """Drop a reversing manoeuvre at the start of a pass.

    A capture that begins in a driveway backs out first: the path goes one
    way for a few metres, stops, and comes back through the same spot. A
    camera replaying that looks the wrong way for the first seconds and
    renders the least-observed part of the scene. Cut at the last heading
    flip (> turn_deg) inside the first head_m metres; returns (path, metres
    dropped)."""
    if len(points) < 4:
        return points, 0.0
    seg = np.diff(points[:, :2], axis=0)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
    cut = 0
    for i in range(1, len(seg)):
        if dist[i] > head_m:
            break
        a, b = seg[i - 1], seg[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        ang = np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0)))
        if ang > turn_deg:
            cut = i
    return points[cut:], float(dist[cut])


def camera_path(points: np.ndarray, height_offset_m: float = 0.0,
                look_ahead: int = 4) -> np.ndarray:
    """camera-to-world matrices along a path, OpenCV axes (x right, y down, z fwd).

    `look_ahead` is in samples; callers size it from the sample spacing so the
    heading is taken over a few metres of road, not a few centimetres."""
    pts = points.copy()
    pts[:, 2] += height_offset_m
    mats = []
    up = np.array([0.0, 0.0, 1.0])          # ENU up
    for i in range(len(pts)):
        j = min(i + look_ahead, len(pts) - 1)
        fwd = pts[j] - pts[i]
        if np.linalg.norm(fwd) < 1e-6:
            fwd = pts[i] - pts[max(i - look_ahead, 0)]
        n = np.linalg.norm(fwd)
        if n < 1e-6:
            fwd = np.array([1.0, 0.0, 0.0])
            n = 1.0
        fwd = fwd / n
        right = np.cross(fwd, up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        down = np.cross(fwd, right)
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, fwd
        c2w[:3, 3] = pts[i]
        mats.append(c2w)
    return np.stack(mats)


def render(chunk: Path, out: Path | None = None, ckpt: Path | None = None,
           corridor: Path | None = None, width: int = 1280, height: int = 720,
           fov_deg: float = 90.0, spacing_m: float = 0.35, fps: int = 30,
           smooth_window: int = 9, height_offset_m: float = 0.0,
           pass_index: int | None = None) -> Path:
    import torch                                   # noqa: PLC0415
    from gsplat import rasterization               # noqa: PLC0415

    from .mesh import _load_splats                 # noqa: PLC0415

    chunk = Path(chunk)
    out = Path(out or chunk / "drive")
    out.mkdir(parents=True, exist_ok=True)
    corridor_path = Path(corridor or chunk / "corridor.json")
    cor = json.loads(corridor_path.read_text())

    passes = cor.get("passes") or cor.get("points") and [{"points": cor["points"]}]
    if not passes:
        raise SystemExit(f"{corridor_path}: no camera path to follow")
    if pass_index is not None:
        passes = [passes[pass_index]]
    else:
        # longest pass first: that is the through-route rather than a stub
        passes = sorted(passes, key=lambda p: -len(p["points"]))
    p0 = passes[0]
    print(f"[drive] corridor source={cor.get('source', 'gps')} pass video="
          f"{p0.get('video')} seq={p0.get('seq_range')} ({len(p0['points'])} pts)", flush=True)

    if ckpt is None:
        ckpts = list((chunk / "splat" / "ckpts").glob("ckpt_*.pt"))
        if not ckpts:
            raise SystemExit(f"no checkpoint under {chunk}/splat/ckpts")
        ckpt = max(ckpts, key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or -1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats = _load_splats(Path(ckpt), device)
    sh_degree = int(round(splats["sh"].shape[1] ** 0.5)) - 1

    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
                     dtype=torch.float32, device=device)

    raw, dropped = trim_reversals(np.asarray(passes[0]["points"], dtype=np.float64))
    if dropped:
        print(f"[drive] skipped a reversing manoeuvre: first {dropped:.0f} m", flush=True)
    # heading over ~3 m of road and smoothing over ~3 m, whatever the frame spacing
    window = max(smooth_window, int(round(3.0 / spacing_m)) | 1)
    look_ahead = max(4, int(round(3.0 / spacing_m)))
    pts = _smooth(_resample(raw, spacing_m), window)
    c2w = camera_path(pts, height_offset_m, look_ahead=look_ahead)
    print(f"[drive] {len(c2w)} frames, {len(pts) * spacing_m:.0f} m at "
          f"{spacing_m * fps * 3.6:.0f} km/h equivalent", flush=True)

    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    import cv2                                     # noqa: PLC0415
    for i, m in enumerate(c2w):
        view = torch.linalg.inv(torch.tensor(m, dtype=torch.float32, device=device))
        with torch.no_grad():
            img, _, _ = rasterization(
                splats["means"], splats["quats"], splats["scales"],
                splats["opacities"], splats["sh"], view[None], K[None],
                width, height, sh_degree=sh_degree, render_mode="RGB")
        rgb = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if (i + 1) % 100 == 0:
            print(f"[drive]   {i + 1}/{len(c2w)}", flush=True)

    video = out / "drive.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "%05d.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(video)], check=True)
    print(f"[drive] wrote {video}")
    return video
