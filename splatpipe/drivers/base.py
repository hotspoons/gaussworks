# SPDX-License-Identifier: Apache-2.0
"""The contract every capture format implements.

A driver knows one projection. It does NOT know about GPS, frame spacing,
chunking, masking, or what the images are for -- those live in the pipeline
stages and stay identical across cameras. Adding a camera whose projection we
already speak is a YAML profile and no code at all; adding a genuinely new
projection is one subclass with three methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..profiles import CameraProfile


class Driver(ABC):
    """Maps ray directions in the rig frame to pixels of a decoded frame set."""

    #: how many decoded images make up one instant of capture
    n_streams: int = 1
    #: True if the pipeline should reframe; False passes frames through
    reframes: bool = True

    def __init__(self, profile: CameraProfile):
        self.profile = profile
        self._ready = False

    # -- container -------------------------------------------------------
    def ffmpeg_maps(self) -> list[str | None]:
        """`-map` selectors for the streams this driver needs, in order."""
        return [f"0:v:{i}" for i in range(self.n_streams)] if self.n_streams > 1 else [None]

    def prepare_sizes(self, sizes: list[tuple[int, int]]) -> None:
        """Learn stream resolutions. Called BEFORE any frame is decoded, from
        the ffprobe the profile match already needed, so view planning knows
        the real lens geometry rather than a guess."""
        self._ready = True

    def prepare(self, frames: list[np.ndarray]) -> None:
        self.prepare_sizes([(f.shape[1], f.shape[0]) for f in frames])

    # -- geometry --------------------------------------------------------
    @property
    def lenses(self):
        return self.profile.lenses

    @abstractmethod
    def half_fov_deg(self, lens: int) -> float:
        """Coverage half-angle of `lens`, degrees from its own optical axis."""

    def usable_half_fov_deg(self, lens: int) -> float:
        """Half-angle we are willing to *use*, which is a little less.

        The outer ring of any wide lens is its worst: lowest angular
        resolution, most flare, most decentring error. A profile can trim it
        explicitly; otherwise we use it all.
        """
        want = self.lenses[lens].usable_half_fov_deg
        return min(want, self.half_fov_deg(lens)) if want else self.half_fov_deg(lens)

    @abstractmethod
    def sample(self, key, dirs: np.ndarray, frames: list[np.ndarray],
               lens: int) -> np.ndarray:
        """Resample `frames` along `dirs` using ONE lens. Uncovered -> black."""

    @abstractmethod
    def coverage(self, key, dirs: np.ndarray, lens: int) -> np.ndarray:
        """Bool mask, same shape as `dirs[...,0]`: does `lens` see this ray?"""

    def sample_sphere(self, key, dirs: np.ndarray,
                      frames: list[np.ndarray]) -> np.ndarray:
        """All lenses blended into one image -- viewing only, never training."""
        return self.sample(key, dirs, frames, 0)
