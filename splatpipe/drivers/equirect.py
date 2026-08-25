# SPDX-License-Identifier: Apache-2.0
"""Pre-stitched equirectangular input: one stream, one (fictional) lens."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Driver


class EquirectDriver(Driver):
    n_streams = 1

    def __init__(self, profile):
        super().__init__(profile)
        self._maps: dict = {}

    def half_fov_deg(self, lens: int) -> float:
        return float(self.lenses[lens].half_fov_deg or 180.0)

    def _grid(self, key, dirs, shape):
        h, w = shape
        ck = (key, w, h)
        if ck not in self._maps:
            lon = np.arctan2(dirs[..., 0], dirs[..., 1])
            lat = np.arcsin(np.clip(dirs[..., 2], -1, 1))
            mx = ((lon / (2 * np.pi) + 0.5) * w).astype(np.float32)
            my = ((0.5 - lat / np.pi) * h).astype(np.float32)
            self._maps[ck] = (mx, my)
        return self._maps[ck]

    def sample(self, key, dirs, frames, lens):
        img = frames[0]
        mx, my = self._grid(key, dirs, img.shape[:2])
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def covers(self, dirs, lens):
        return np.ones(dirs.shape[:-1], bool)
