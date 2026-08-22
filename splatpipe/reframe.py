# SPDX-License-Identifier: Apache-2.0
"""Equirectangular -> virtual pinhole reframing.

A 360 frame becomes K ideal-pinhole views (no distortion), which COLMAP
models exactly (PINHOLE) and every splat trainer consumes without an
undistortion pass. Remap grids are cached per (equirect size, view).
"""

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_VIEWS = [
    {"yaw": 0, "pitch": -10, "fov": 100, "width": 1600, "height": 1200},
    {"yaw": 90, "pitch": -10, "fov": 100, "width": 1600, "height": 1200},
    {"yaw": 180, "pitch": -10, "fov": 100, "width": 1600, "height": 1200},
    {"yaw": 270, "pitch": -10, "fov": 100, "width": 1600, "height": 1200},
]


@dataclass(frozen=True)
class View:
    yaw: float      # degrees, 0 = video forward, clockwise from above
    pitch: float    # degrees, negative = down
    fov: float      # horizontal, degrees
    width: int
    height: int


def build_maps(eq_w: int, eq_h: int, view: View) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap grids sampling an equirect image into a pinhole view."""
    f = 0.5 * view.width / np.tan(np.radians(view.fov) / 2)
    xs = np.arange(view.width, dtype=np.float64) - (view.width - 1) / 2
    ys = np.arange(view.height, dtype=np.float64) - (view.height - 1) / 2
    xv, yv = np.meshgrid(xs, ys)
    d = np.stack([xv, yv, np.full_like(xv, f)], axis=-1)  # x right, y down, z fwd
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    p, y = np.radians(view.pitch), np.radians(view.yaw)
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    d = d @ (ry @ rx).T

    lon = np.arctan2(d[..., 0], d[..., 2])   # 0 at equirect center
    lat = np.arcsin(np.clip(-d[..., 1], -1, 1))
    map_x = ((lon / (2 * np.pi) + 0.5) * eq_w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * eq_h).astype(np.float32)
    return map_x, map_y


class Reframer:
    def __init__(self, views: list[dict] | None = None):
        self.views = [View(**v) for v in (views or DEFAULT_VIEWS)]
        self._maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, equirect_bgr: np.ndarray) -> list[np.ndarray]:
        eq_h, eq_w = equirect_bgr.shape[:2]
        out = []
        for view in self.views:
            key = (eq_w, eq_h, view)
            if key not in self._maps:
                self._maps[key] = build_maps(eq_w, eq_h, view)
            mx, my = self._maps[key]
            out.append(cv2.remap(equirect_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
        return out
