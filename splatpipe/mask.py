# SPDX-License-Identifier: Apache-2.0
"""Stage 1b: mask the rig out of every frame.

The capture vehicle is rigid relative to the camera, so it lands on the same
pixels in every single frame. That is the worst kind of occluder -- COLMAP
happily matches features on your own roof and pulls poses toward it, and the
trainer spends capacity on a "surface" that exists at a different world
position in every image -- and simultaneously the easiest to remove: one static
mask per camera, no per-frame segmentation, no model.

Finding it needs no annotation either. Take the temporal median over many
frames: world content averages into mush while anything bolted to the car keeps
its edges, so edge energy in the median is a static-edge detector.

We locate the vehicle's SILHOUETTE rather than its interior, because a glossy
roof is a poor target for staticness tests -- moving reflections make the glass
itself look dynamic even though its outline never moves. So per column: find the
topmost strong static edge in the lower part of the frame, smooth that boundary
across columns, and mask everything below it. Works for a dark or a light
vehicle, since it keys on the edge, not the colour.

Masks are written in COLMAP's convention (`<image path>.png`, black = ignore),
one real PNG per camera with a link per frame.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from .reframe import DEFAULT_VIEWS


def seam_mask(view: dict, band_deg: float = 6.0) -> np.ndarray:
    """Mask the lens boundary out of one virtual view.

    A .360 is two lenses, and every direction within a few degrees of the
    plane between them exists twice in the file -- once per lens, from optical
    centres ~3 cm apart and through different calibration. Those pixels
    disagree geometrically, and a reconstruction resolves the disagreement by
    growing two of everything near the seam. Since the rig is fixed and the
    virtual views are at fixed angles, the boundary lands on the SAME pixels in
    every frame, so a static mask removes the ambiguity for good.

    Cheaper and more honest than pretending we can stitch: we drop the pixels
    we cannot trust rather than blending two versions of the truth.
    """
    from .eac import view_dirs                     # noqa: PLC0415
    d = view_dirs(view["width"], view["height"], view["fov"],
                  view["yaw"], view["pitch"])
    # lens axes are +-Y in this convention, so the boundary is |dir . Y| ~ 0
    near_seam = np.abs(d[..., 1]) < np.sin(np.radians(band_deg))
    return np.where(near_seam, 0, 255).astype(np.uint8)


def _local_contrast(gray: np.ndarray, k: int = 7) -> np.ndarray:
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    sq = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def staticness(paths: list[Path], scale: int = 4) -> np.ndarray:
    """Per-pixel [0,1]-ish score: how much detail survives the temporal median."""
    frames, contrasts = [], None
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        small = cv2.resize(im, (im.shape[1] // scale, im.shape[0] // scale))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames.append(small)
        lc = _local_contrast(gray)
        contrasts = lc if contrasts is None else contrasts + lc
    if not frames:
        raise SystemExit("no frames to analyse")
    med = np.median(np.stack(frames), 0).astype(np.uint8)
    lc_med = _local_contrast(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY))
    lc_frames = contrasts / len(frames)
    return lc_med / (lc_frames + 1e-3)


def static_edges(paths: list[Path], scale: int = 4) -> np.ndarray:
    """Edge energy of the temporal median: strong only where nothing moves."""
    frames = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        frames.append(cv2.resize(im, (im.shape[1] // scale, im.shape[0] // scale)))
    if not frames:
        raise SystemExit("no frames to analyse")
    med = np.median(np.stack(frames), 0).astype(np.uint8)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    return cv2.GaussianBlur(lap, (0, 0), 2.0)


def vehicle_mask(paths: list[Path], shape: tuple[int, int], scale: int = 4,
                 search_from: float = 0.35, dark_pct: float = 45.0,
                 smooth_cols: int = 41, pad_px: int = 8) -> np.ndarray:
    """Binary mask: 0 where the rig is, 255 where the world is.

    Keys on the temporal median being much darker over the vehicle than over
    the scene, which holds for a car roof / bonnet / tinted glass and is far
    more stable than any edge or staticness test on reflective paint. Then per
    column it takes the TOPMOST vehicle pixel and fills everything below, so a
    bright reflection cannot punch a hole through the middle of the mask.

    `search_from` bounds how far up the frame the rig may reach, so a dark
    treeline can never be mistaken for it. If a rig defeats this (a white car
    under a bright sky), drop a hand-drawn `masks/<cam>/_mask.png` in place --
    the geometry is fixed, so one polygon lasts the whole campaign.
    """
    frames = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        frames.append(cv2.resize(im, (im.shape[1] // scale, im.shape[0] // scale)))
    if not frames:
        raise SystemExit("no frames to analyse")
    med = np.median(np.stack(frames), 0).astype(np.uint8)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lo = int(h * search_from)

    thr = np.percentile(gray[lo:, :], dark_pct)
    dark = np.zeros((h, w), np.uint8)
    dark[lo:, :] = (gray[lo:, :] < thr).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # the rig is the dark region attached to the bottom edge
    _, labels = cv2.connectedComponents(dark)
    keep = np.zeros_like(dark)
    for lab in np.unique(labels[-1, :]):
        if lab != 0:
            keep[labels == lab] = 1

    # per-column boundary, smoothed, then filled downward
    boundary = np.full(w, h, np.float32)
    for c in range(w):
        rows = np.flatnonzero(keep[:, c])
        if len(rows):
            boundary[c] = rows[0]
    k = max(smooth_cols | 1, 3)
    boundary = cv2.medianBlur(boundary.reshape(1, -1).astype(np.float32), 1).ravel()
    boundary = cv2.blur(boundary.reshape(1, -1), (k, 1)).ravel()

    mask = np.full((h, w), 255, np.uint8)
    for c in range(w):
        top = int(max(boundary[c] - pad_px, lo * 0.5))
        if top < h:
            mask[top:, c] = 0
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def build(frames_dir: Path, sample: int = 60, search_from: float = 0.35,
          dark_pct: float = 45.0, seam_band_deg: float = 0.0,
          views: list[dict] | None = None) -> Path:
    frames_dir = Path(frames_dir)
    images = frames_dir / "images"
    out = frames_dir / "masks"
    out.mkdir(exist_ok=True)
    report = {}

    for cam in sorted(p for p in images.iterdir() if p.is_dir()):
        files = sorted(cam.glob("*.jpg"))
        if not files:
            continue
        step = max(len(files) // sample, 1)
        probe = files[::step][:sample]
        h, w = cv2.imread(str(files[0])).shape[:2]
        shared_existing = out / cam.name / "_mask.png"
        if shared_existing.exists() and shared_existing.stat().st_size > 0 \
                and os.environ.get("GAUSSWORKS_KEEP_MASKS"):
            print(f"[mask] {cam.name}: keeping existing mask")
            continue
        mask = vehicle_mask(probe, (h, w), search_from=search_from,
                            dark_pct=dark_pct)
        if seam_band_deg > 0:
            specs = views or DEFAULT_VIEWS
            k = int(cam.name.replace("cam", "")) if cam.name[3:].isdigit() else 0
            if k < len(specs):
                sm = seam_mask(specs[k], seam_band_deg)
                if sm.shape != mask.shape:
                    sm = cv2.resize(sm, (mask.shape[1], mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                mask = np.minimum(mask, sm)
        covered = float((mask == 0).mean())

        cam_out = out / cam.name
        cam_out.mkdir(parents=True, exist_ok=True)
        shared = cam_out / "_mask.png"
        cv2.imwrite(str(shared), mask)
        # COLMAP wants <image>.png alongside; identical per camera, so link
        for f in files:
            link = cam_out / (f.name + ".png")
            if not link.exists():
                os.symlink(shared.name, link)
        report[cam.name] = round(covered, 4)
        print(f"[mask] {cam.name}: rig covers {covered:.1%} of frame "
              f"({len(probe)} frames sampled)")

    (out / "masks.json").write_text(json.dumps(
        {"masked_fraction": report, "search_from": search_from,
         "dark_pct": dark_pct}, indent=1))
    return out
