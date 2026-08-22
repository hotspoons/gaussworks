# SPDX-License-Identifier: Apache-2.0
"""GoPro MAX .360 EAC (equi-angular cubemap) sampling.

Attribution: the .360 track/face layout, split-half storage, and blend-strip
arithmetic are ported from trek-view/max2sphere (Apache-2.0, Copyright Trek
View; based on Paul Bourke's max2sphere) — https://github.com/trek-view/max2sphere
and Trek View's reverse-engineering write-ups (trekview.org). This file is an
independent numpy re-derivation: faces are modeled as an explicit per-face
basis table and sampled direction->pixel in one remap; per-face orientations
were re-validated empirically on real footage (see git history / README).
See THIRD_PARTY.md for the full third-party inventory.

A .360 carries two video tracks, each a horizontal strip of three cube faces:

    track 1: [ left-half-pair | FRONT | right-half-pair ]
    track 2: [ down-half-pair | BACK  | up-half-pair ]   (faces rotated 90 CCW)

FRONT/BACK occupy the strip's center at full width (= face size). The four
side faces are stored as two halves with a `blend`-px duplicated strip between
them. Faces use the equi-angular mapping (q = atan(p) * 4/pi per axis).

Math ported from trek-view/max2sphere (Paul Bourke); we build cv2.remap grids
from arbitrary ray directions straight to track pixels, so 360 -> pinhole is a
single resample (no equirect intermediate). In blend strips we snap to the
nearer half instead of alpha-blending: worst case a sub-pixel seam on 64 of
1376 columns, irrelevant for reconstruction.

Known templates (track WxH): 4096x1344 (5.6K) and 2272x736 (3K). Other sizes
(e.g. Max 2 8K) are derived from the same proportions and worth eyeballing
with eac_to_equirect() before trusting.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# faces
LEFT, RIGHT, TOP, DOWN, FRONT, BACK = range(6)
TRACK2_FACES = (DOWN, BACK, TOP)  # stored in the second video track


@dataclass(frozen=True)
class Template:
    width: int       # track width
    height: int      # track height == face size
    side: int        # width of each side (half-pair) section
    blend: int       # duplicated strip between the halves

    @classmethod
    def for_size(cls, w: int, h: int) -> "Template":
        known = {(4096, 1344): 32, (2272, 736): 16}
        side = (w - h) // 2
        if (w, h) not in known:
            print(f"[eac] warning: unknown track size {w}x{h}, deriving template")
        return cls(w, h, side, known.get((w, h), side - h))


# Per-face storage basis, world axes: x right, y forward, z up.
# C: face center direction; U: world direction along increasing stored columns;
# V: along increasing stored rows. Derived from the strips being continuous
# panoramas: track 1 sweeps the horizon (left->front->right, stored upright),
# track 2 sweeps a vertical great circle (floor->back->ceiling, stored
# rotated 90deg). V_ROWS2 (rows of the rotated track-2 faces) was determined
# empirically on real footage via seam-continuity scoring; note every face
# then satisfies U x V = C, one coherent right-handed convention.
_X, _Y, _Z = np.eye(3)
V_ROWS2 = -_X
FACE_BASIS = {
    LEFT:  (-_X, +_Y, -_Z),
    FRONT: (+_Y, +_X, -_Z),
    RIGHT: (+_X, -_Y, -_Z),
    DOWN:  (-_Z, -_Y, V_ROWS2),
    BACK:  (-_Y, +_Z, V_ROWS2),
    TOP:   (+_Z, +_Y, V_ROWS2),
}
# (track, section): section 0 = left half-pair, 1 = center face, 2 = right half-pair
FACE_SLOT = {LEFT: (0, 0), FRONT: (0, 1), RIGHT: (0, 2),
             DOWN: (1, 0), BACK: (1, 1), TOP: (1, 2)}


def eac_maps(dirs: np.ndarray, tpl: Template):
    """Directions (...,3) -> (track_index, map_x, map_y) in track pixels."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.where(ay >= np.maximum(ax, az), np.where(y >= 0, FRONT, BACK),
                    np.where(ax >= az, np.where(x >= 0, RIGHT, LEFT),
                             np.where(z >= 0, TOP, DOWN)))

    k = 4.0 / np.pi
    track = np.zeros(face.shape, np.uint8)
    ix = np.empty(face.shape, np.float64)
    iy = np.empty(face.shape, np.float64)
    duv = tpl.blend / tpl.side

    for f, (C, U, V) in FACE_BASIS.items():
        m = face == f
        if not m.any():
            continue
        dc = np.maximum(x[m] * C[0] + y[m] * C[1] + z[m] * C[2], 1e-12)
        # equi-angular face params in [0,1], stored-pixel orientation
        u = np.clip((np.arctan((x[m] * U[0] + y[m] * U[1] + z[m] * U[2]) / dc) * k + 1) * 0.5,
                    0, 1 - 1e-7)
        v = np.clip((np.arctan((x[m] * V[0] + y[m] * V[1] + z[m] * V[2]) / dc) * k + 1) * 0.5,
                    0, 1 - 1e-7)
        trk, section = FACE_SLOT[f]
        track[m] = trk
        iy[m] = v * tpl.height
        if section == 1:
            ix[m] = tpl.side + u * tpl.height
        else:
            # side sections store the face as two halves around a duplicated
            # blend strip; snap to the nearer half (sub-pixel seam at worst)
            u_half = np.where(u < 0.5,
                              2 * (0.5 - duv) * u,
                              2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv)
            x0 = 0 if section == 0 else tpl.side + tpl.height
            ix[m] = x0 + u_half * tpl.side

    return track, ix.astype(np.float32), iy.astype(np.float32)


class EacSampler:
    """Resamples (track1, track2) frame pairs into pinhole views (or equirect)."""

    def __init__(self, track_w: int, track_h: int):
        self.tpl = Template.for_size(track_w, track_h)
        self._cache: dict = {}

    def _maps(self, key, dirs):
        if key not in self._cache:
            self._cache[key] = eac_maps(dirs, self.tpl)
        return self._cache[key]

    def sample(self, key, dirs, track1: np.ndarray, track2: np.ndarray) -> np.ndarray:
        track, mx, my = self._maps(key, dirs)
        a = cv2.remap(track1, mx, my, cv2.INTER_LINEAR)
        b = cv2.remap(track2, mx, my, cv2.INTER_LINEAR)
        return np.where(track[..., None] == 0, a, b)


def view_dirs(width: int, height: int, fov_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Pinhole ray grid in max2sphere convention (x right, y fwd, z up)."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    xs = np.arange(width, dtype=np.float64) - (width - 1) / 2
    ys = np.arange(height, dtype=np.float64) - (height - 1) / 2
    xv, yv = np.meshgrid(xs, ys)
    # camera frame: x right, y down, z forward -> world: x right, y fwd, z up
    d = np.stack([xv, np.full_like(xv, f), -yv], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(yw), np.sin(yw), 0], [-np.sin(yw), np.cos(yw), 0], [0, 0, 1]])
    return d @ (rz @ rx).T


def equirect_dirs(width: int) -> np.ndarray:
    height = width // 2
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lon, lat)
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), cl * np.cos(lon), np.sin(lat)], axis=-1)


def eac_to_equirect(track1: np.ndarray, track2: np.ndarray, width: int = 4096) -> np.ndarray:
    """Debug/eyeball helper: full equirect from one EAC frame pair."""
    sampler = EacSampler(track1.shape[1], track1.shape[0])
    return sampler.sample(("equirect", width), equirect_dirs(width), track1, track2)
