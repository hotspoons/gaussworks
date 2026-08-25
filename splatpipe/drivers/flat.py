# SPDX-License-Identifier: Apache-2.0
"""Ordinary rectilinear camera: nothing to reproject."""

from __future__ import annotations

import numpy as np

from .base import Driver


class FlatDriver(Driver):
    n_streams = 1
    reframes = False

    def half_fov_deg(self, lens: int) -> float:
        return float(self.lenses[lens].half_fov_deg or 60.0)

    def sample(self, key, dirs, frames, lens):
        return frames[0]

    def coverage(self, key, dirs, lens):
        return np.ones(dirs.shape[:-1], bool)
