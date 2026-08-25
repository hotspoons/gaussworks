# SPDX-License-Identifier: Apache-2.0
"""GoPro .360 equi-angular cubemap, two video tracks."""

from __future__ import annotations

import numpy as np

from ..eac import EacSampler
from .base import Driver


class GoProEacDriver(Driver):
    n_streams = 2

    def __init__(self, profile):
        super().__init__(profile)
        self.sampler: EacSampler | None = None

    def _blend_px(self, w: int, h: int) -> int | None:
        g = self.profile.geometry or {}
        by_track = g.get("blend_px_by_track") or {}
        if f"{w}x{h}" in by_track:
            return int(by_track[f"{w}x{h}"])
        return None if g.get("blend_px") is None else int(g["blend_px"])

    def prepare_sizes(self, sizes):
        w, h = sizes[0]
        self.sampler = EacSampler(w, h, self._blend_px(w, h))
        tpl = self.sampler.tpl
        print(f"[eac] {w}x{h}: side={tpl.side} face={tpl.height} "
              f"blend={tpl.blend}px -> each lens sees "
              f"{90 + tpl.overlap_half_deg:.2f} deg from its axis")
        self._ready = True

    def half_fov_deg(self, lens: int) -> float:
        declared = self.lenses[lens].half_fov_deg
        if declared:
            return float(declared)
        if self.sampler is None:
            # planning happens before the first frame is decoded; the templates
            # we ship all land here, and an unknown one is close enough to plan
            # with and gets corrected once prepare() sees the real track size
            return 92.0
        return 90.0 + self.sampler.tpl.overlap_half_deg

    def sample(self, key, dirs, frames, lens):
        return self.sampler.sample(key, dirs, frames[0], frames[1], lens=lens)

    def coverage(self, key, dirs, lens):
        return self.sampler.coverage(key, dirs, lens)

    def sample_sphere(self, key, dirs, frames):
        return self.sampler.sample(key, dirs, frames[0], frames[1], lens=None)
