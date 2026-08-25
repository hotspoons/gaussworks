# SPDX-License-Identifier: Apache-2.0
"""Raw dual-fisheye: two circular images, usually side by side in one stream.

This is the Insta360 X-series shape (and Ricoh Theta, and the GoPro Fusion's
two files). It is the *good* input format for reconstruction, because nothing
has been stitched yet -- the two optical centres are still two optical centres.

Everything camera-specific is in the profile's `geometry.circles`: where each
image circle sits, how big it is, how much of the sphere it spans, and how it
is rolled. Fitting those four numbers per lens is the entire calibration, and
`splatpipe verify` renders the overlay to fit them against.

The projection model maps the angle from the optical axis to a radius:

    equidistant   r = f * theta          (most action cams claim this)
    equisolid     r = 2f * sin(theta/2)
    stereographic r = 2f * tan(theta/2)
    orthographic  r = f * sin(theta)

with f set so that theta = fov/2 lands exactly on `radius`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Driver

_MODELS = {
    "equidistant":   lambda th: th,
    "equisolid":     lambda th: 2.0 * np.sin(th / 2.0),
    "stereographic": lambda th: 2.0 * np.tan(np.minimum(th, np.pi * 0.499) / 2.0),
    "orthographic":  lambda th: np.sin(np.minimum(th, np.pi / 2)),
}


def _frame(axis: np.ndarray, roll_deg: float):
    """Right/down basis of a lens' image plane, given its axis and roll."""
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, axis)
    right /= np.linalg.norm(right)
    down = np.cross(axis, right)
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    return c * right + s * down, -s * right + c * down


class DualFisheyeDriver(Driver):
    n_streams = 1

    def __init__(self, profile):
        super().__init__(profile)
        g = profile.geometry or {}
        self.circles = list(g.get("circles") or [])
        self.model = _MODELS[g.get("model", "equidistant")]
        if len(self.circles) != len(profile.lenses):
            raise SystemExit(f"[{profile.name}] geometry.circles has "
                             f"{len(self.circles)} entries but {len(profile.lenses)} "
                             f"lenses; they must correspond one to one")
        self._maps: dict = {}

    def half_fov_deg(self, lens: int) -> float:
        declared = self.lenses[lens].half_fov_deg
        return float(declared) if declared else float(self.circles[lens]["fov_deg"]) / 2.0

    def _grid(self, key, dirs, lens):
        ck = (key, lens)
        if ck in self._maps:
            return self._maps[ck]
        c = self.circles[lens]
        axis = self.lenses[lens].axis_np
        right, down = _frame(axis, float(c.get("roll_deg", 0.0)))
        half = np.radians(float(c["fov_deg"]) / 2.0)
        f = float(c["radius"]) / self.model(np.array(half))

        cos_t = np.clip(dirs @ axis, -1.0, 1.0)
        theta = np.arccos(cos_t)
        px, py = dirs @ right, dirs @ down
        phi = np.arctan2(py, px)
        r = f * self.model(theta)
        mx = (float(c["cx"]) + r * np.cos(phi)).astype(np.float32)
        my = (float(c["cy"]) + r * np.sin(phi)).astype(np.float32)
        valid = theta <= half
        self._maps[ck] = (mx, my, valid)
        return self._maps[ck]

    def sample(self, key, dirs, frames, lens):
        mx, my, valid = self._grid(key, dirs, lens)
        img = cv2.remap(frames[0], mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.where(valid[..., None], img, 0).astype(np.uint8)

    def coverage(self, key, dirs, lens):
        return self._grid(key, dirs, lens)[2]

    def sample_sphere(self, key, dirs, frames):
        """Nearest-lens composite. Deliberately NOT feathered: a hard edge is
        an honest report of where the viewpoint changes, and this path exists
        for eyeballing geometry, not for making a pretty panorama."""
        out = None
        best = None
        for i in range(len(self.circles)):
            mx, my, valid = self._grid(key, dirs, i)
            cos_t = dirs @ self.lenses[i].axis_np
            img = cv2.remap(frames[0], mx, my, cv2.INTER_LINEAR)
            if out is None:
                out, best = np.zeros_like(img), np.full(dirs.shape[:-1], -2.0)
            take = valid & (cos_t > best)
            out = np.where(take[..., None], img, out)
            best = np.where(take, cos_t, best)
        return out.astype(np.uint8)
