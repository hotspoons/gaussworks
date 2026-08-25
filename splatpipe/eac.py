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

THE HALF BOUNDARY IS A LENS BOUNDARY. FRONT comes from one lens, BACK from the
other, and every side face is stored as two halves -- one per lens -- meeting
in a strip both lenses saw. So the boundary is not a compression artifact or a
packing detail: it is the point where the viewpoint physically changes, by the
few centimetres between the two entrance pupils.

That leaves exactly two honest options, and one dishonest one:

  * sample ONE lens (`lens=0|1`) and report where it has no data. Every output
    pixel then comes from a single real optical centre. This is what ingest
    does, and it is why `eac_maps` returns a `valid` mask.
  * cross-fade the two lenses across the overlap (`lens=None`). Cheap, fine
    for a flythrough or a contact sheet, WRONG as reconstruction input: near
    objects appear twice, and a splat trainer duly grows two of them.
  * warp one lens onto the other until the parallax cancels (what the camera's
    own stitcher does). Best-looking of the three, and the worst input of the
    three, because it destroys the disparity that carries the depth.

Known templates (track WxH): 4096x1344 (5.6K), 2272x736 (3K), 5952x1920 (Max 2
8K). Sizes and blend widths come from the camera profile; anything unlisted is
derived from the proportions and worth eyeballing with `splatpipe verify`.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# faces
LEFT, RIGHT, TOP, DOWN, FRONT, BACK = range(6)
TRACK2_FACES = (DOWN, BACK, TOP)  # stored in the second video track

KNOWN_BLEND = {(4096, 1344): 32, (2272, 736): 16, (5952, 1920): 96}


@dataclass(frozen=True)
class Template:
    width: int       # track width
    height: int      # track height == face size
    side: int        # width of each side (half-pair) section
    blend: int       # duplicated strip between the halves

    @classmethod
    def for_size(cls, w: int, h: int, blend: int | None = None) -> "Template":
        side = (w - h) // 2
        if blend is None:
            blend = KNOWN_BLEND.get((w, h))
            if blend is None:
                blend = side - h
                print(f"[eac] track {w}x{h} not in the profile: deriving "
                      f"blend={blend}px. Check it with `splatpipe verify`.")
        return cls(w, h, side, blend)

    @property
    def duv(self) -> float:
        """Overlap width as a fraction of a side section."""
        return self.blend / self.side

    @property
    def u_a_max(self) -> float:
        """Largest face-u the low-u (first) half still holds pixels for."""
        return 0.5 / (2 * (0.5 - self.duv))

    @property
    def u_b_min(self) -> float:
        """Smallest face-u the high-u (second) half still holds pixels for."""
        return 1.0 - self.u_a_max

    @property
    def overlap_half_deg(self) -> float:
        """How far past 90 deg from its own axis each lens sees, in degrees.

        Face-u is equi-angular over +-45 deg about the face centre, and a side
        face is centred exactly on the lens boundary, so the half that reaches
        u = u_a_max reaches (2*u_a_max - 1) * 45 degrees past it.
        """
        return (2 * self.u_a_max - 1) * 45.0


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

FRONT_AXIS = np.array([0.0, 1.0, 0.0])   # lens 0 looks along +y in the rig frame


@dataclass
class EacMap:
    """Remap grids from a direction grid into a (track1, track2) frame pair."""
    track: np.ndarray        # 0 -> sample track1, 1 -> track2
    ix: np.ndarray           # primary column
    iy: np.ndarray           # row (shared)
    ix2: np.ndarray          # secondary column (other lens' half), blend mode
    weight: np.ndarray       # 0..1 mix toward ix2, blend mode; all-zero otherwise
    valid: np.ndarray        # bool: this lens actually holds a pixel here

    def __iter__(self):      # legacy tuple unpacking (eac_align)
        return iter((self.track, self.ix, self.iy, self.ix2, self.weight))


def eac_maps(dirs: np.ndarray, tpl: Template, lens: int | None = None) -> EacMap:
    """Directions (...,3) -> remap grids.

    `lens=None` cross-fades the two lenses across the overlap (viewing).
    `lens=0|1` samples that lens alone and marks everything it cannot see
    invalid (reconstruction) -- see the module docstring for why that matters.
    """
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.where(ay >= np.maximum(ax, az), np.where(y >= 0, FRONT, BACK),
                    np.where(ax >= az, np.where(x >= 0, RIGHT, LEFT),
                             np.where(z >= 0, TOP, DOWN)))

    k = 4.0 / np.pi
    track = np.zeros(face.shape, np.uint8)
    ix = np.empty(face.shape, np.float64)
    ix2 = np.empty(face.shape, np.float64)
    weight = np.zeros(face.shape, np.float64)
    iy = np.empty(face.shape, np.float64)
    valid = np.ones(face.shape, bool)
    duv = tpl.duv

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
            # a centre face lies wholly inside one lens' hemisphere
            owner = 0 if float(C @ FRONT_AXIS) > 0 else 1
            ix[m] = tpl.side + u * tpl.height
            ix2[m] = ix[m]
            if lens is not None and lens != owner:
                valid[m] = False
            continue

        # Each side face is two half-images, one per lens, overlapping in a
        # `blend`-wide strip. u_a spans the first (low-column) half, u_b the
        # second; each is only real where it lands inside its own half.
        x0 = 0 if section == 0 else tpl.side + tpl.height
        u_a = 2 * (0.5 - duv) * u
        u_b = 2 * (0.5 - duv) * (u - 0.5) + 0.5 + duv
        # which half belongs to the front lens: the one u grows toward
        front_is_b = float(U @ FRONT_AXIS) > 0
        lens_a = 1 if front_is_b else 0

        if lens is None:
            lo, hi = 0.5 - 2 * duv, 0.5 + 2 * duv
            in_a = u_a <= lo
            in_b = u_b >= hi
            w = np.clip((u_a - lo) / max(2 * duv, 1e-9), 0.0, 1.0)
            w = np.where(in_a, 0.0, np.where(in_b, 1.0, w))
            ix[m] = x0 + np.where(in_b, u_b, u_a) * tpl.side
            ix2[m] = x0 + u_b * tpl.side
            weight[m] = np.where(in_a | in_b, 0.0, w)
        elif lens == lens_a:
            ix[m] = x0 + u_a * tpl.side
            ix2[m] = ix[m]
            valid[m] = u <= tpl.u_a_max
        else:
            ix[m] = x0 + u_b * tpl.side
            ix2[m] = ix[m]
            valid[m] = u >= tpl.u_b_min

    return EacMap(track, ix.astype(np.float32), iy.astype(np.float32),
                  ix2.astype(np.float32), weight.astype(np.float32), valid)


class EacSampler:
    """Resamples (track1, track2) frame pairs into pinhole views (or equirect).

    Pass `lens=` to `sample` for single-lens output (and use `coverage` to get
    the matching validity mask); omit it for a cross-faded full sphere.
    """

    def __init__(self, track_w: int, track_h: int, blend: int | None = None):
        self.tpl = Template.for_size(track_w, track_h, blend)
        self._cache: dict = {}

    def _maps(self, key, dirs, lens) -> EacMap:
        ck = (key, lens)
        if ck not in self._cache:
            self._cache[ck] = eac_maps(dirs, self.tpl, lens=lens)
        return self._cache[ck]

    def coverage(self, key, dirs, lens: int) -> np.ndarray:
        """Bool mask: where this lens holds real pixels for these directions."""
        return self._maps(key, dirs, lens).valid

    def sample(self, key, dirs, track1: np.ndarray, track2: np.ndarray,
               lens: int | None = None) -> np.ndarray:
        mp = self._maps(key, dirs, lens)

        def pick(col):
            a = cv2.remap(track1, col, mp.iy, cv2.INTER_LINEAR)
            b = cv2.remap(track2, col, mp.iy, cv2.INTER_LINEAR)
            return np.where(mp.track[..., None] == 0, a, b)

        primary = pick(mp.ix)
        if lens is not None:
            return np.where(mp.valid[..., None], primary, 0).astype(np.uint8)
        if not mp.weight.any():
            return primary.astype(np.uint8)
        secondary = pick(mp.ix2).astype(np.float32)
        a = mp.weight[..., None]
        return ((1.0 - a) * primary.astype(np.float32)
                + a * secondary).clip(0, 255).astype(np.uint8)


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


def eac_to_equirect(track1: np.ndarray, track2: np.ndarray, width: int = 4096,
                    blend: int | None = None, lens: int | None = None) -> np.ndarray:
    """Debug/eyeball helper: full equirect from one EAC frame pair."""
    sampler = EacSampler(track1.shape[1], track1.shape[0], blend)
    return sampler.sample(("equirect", width), equirect_dirs(width), track1, track2,
                          lens=lens)
