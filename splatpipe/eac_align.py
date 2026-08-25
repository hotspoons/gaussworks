# SPDX-License-Identifier: Apache-2.0
"""Estimate the rotation that aligns a .360's two lens hemispheres.

A .360 stores what the two lenses saw, not a stitched sphere, and the lenses
are never mounted exactly where the ideal geometry assumes: there is a small
rotation between them. GoPro's own software corrects it with per-camera
calibration. Decoding with ideal geometry instead leaves the same object at
two slightly different directions depending on which lens saw it, so the
overlap ghosts -- and a reconstruction fed those frames grows two of
everything near the seam (a parked truck appearing twice, for example).

We recover that rotation from the footage itself: sample a band of directions
straddling the lens boundary, render it once from each lens, and search for
the small rotation of the second lens that maximises normalised correlation
between them. One number per camera body, stable for the whole capture.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .eac import TRACK2_FACES, EacSampler, Template, eac_maps


def _rot(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx, sy, cy, sz, cz = (math.sin(rx), math.cos(rx), math.sin(ry),
                              math.cos(ry), math.sin(rz), math.cos(rz))
    return (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))


def _band_dirs(n_theta: int = 900, n_phi: int = 60, half_width_deg: float = 12.0):
    """Directions in a band straddling the lens boundary.

    The boundary is the great circle perpendicular to the lens axis; with the
    lenses on +-Y that is the XZ plane, so we sweep around it and step a few
    degrees either side.
    """
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    phi = np.radians(np.linspace(-half_width_deg, half_width_deg, n_phi))
    t, p = np.meshgrid(theta, phi)
    # start on the XZ great circle, tilt by phi towards +-Y
    return np.stack([np.cos(t) * np.cos(p), np.sin(p), np.sin(t) * np.cos(p)], axis=-1)


def _sample_track(track_img, dirs, tpl, want_track: int):
    trk, mx, my, _, _ = eac_maps(dirs, tpl)
    img = cv2.remap(track_img, mx, my, cv2.INTER_LINEAR)
    return img, (trk == want_track)


def estimate(track1: np.ndarray, track2: np.ndarray, coarse_deg: float = 2.0,
             steps: int = 5, refine_rounds: int = 3) -> dict:
    """-> {'rx','ry','rz'} radians rotating lens 2 into lens 1's frame."""
    tpl = Template.for_size(track1.shape[1], track1.shape[0])
    dirs = _band_dirs()
    g1 = cv2.cvtColor(track1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(track2, cv2.COLOR_BGR2GRAY)

    ref, m1 = _sample_track(g1, dirs, tpl, 0)
    best = np.zeros(3)
    span = math.radians(coarse_deg)

    def score(r):
        rotated = dirs @ _rot(*r).T
        cmp_, m2 = _sample_track(g2, rotated, tpl, 1)
        both = m1 & m2
        if both.sum() < 500:
            return -1.0
        a = ref[both].astype(np.float32)
        b = cmp_[both].astype(np.float32)
        a -= a.mean(); b -= b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / denom) if denom > 1e-6 else -1.0

    base = score(best)
    for _ in range(refine_rounds):
        grid = np.linspace(-span, span, steps)
        improved = False
        for axis in range(3):
            vals = []
            for d in grid:
                trial = best.copy()
                trial[axis] += d
                vals.append((score(trial), d))
            s, d = max(vals)
            if d != 0.0 and s > base:
                best[axis] += d
                base = s
                improved = True
        span /= 2.5
        if not improved and span < math.radians(0.05):
            break

    return {"rx": float(best[0]), "ry": float(best[1]), "rz": float(best[2]),
            "ncc": round(base, 4),
            "degrees": [round(math.degrees(v), 4) for v in best]}


def calibrate(video: Path, out: Path | None = None, at_s: float = 120.0,
              frames: int = 3) -> dict:
    """Average an estimate over a few frames and write eac_align.json."""
    import subprocess                              # noqa: PLC0415
    import tempfile                                # noqa: PLC0415

    video = Path(video)
    out = Path(out or video.parent / "eac_align.json")
    ests = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(frames):
            t = at_s + i * 30
            for trk, name in ((0, "a"), (1, "b")):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t),
                                "-i", str(video), "-map", f"0:v:{trk}",
                                "-frames:v", "1", f"{td}/{name}.png"], check=True)
            t1, t2 = cv2.imread(f"{td}/a.png"), cv2.imread(f"{td}/b.png")
            est = estimate(t1, t2)
            print(f"[align] t={t:.0f}s  ncc={est['ncc']:.4f}  "
                  f"deg={est['degrees']}", flush=True)
            ests.append(est)
    avg = {k: float(np.mean([e[k] for e in ests])) for k in ("rx", "ry", "rz")}
    avg["ncc"] = float(np.mean([e["ncc"] for e in ests]))
    avg["degrees"] = [round(math.degrees(avg[k]), 4) for k in ("rx", "ry", "rz")]
    avg["frames"] = len(ests)
    out.write_text(json.dumps(avg, indent=1))
    print(f"[align] wrote {out}: {avg['degrees']} deg, ncc {avg['ncc']:.4f}")
    return avg
