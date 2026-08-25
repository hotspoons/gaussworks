# SPDX-License-Identifier: Apache-2.0
"""Camera profiles: everything hardware-specific, out of the pipeline code.

A profile answers four questions about a capture device, and nothing else:

  1. how do I recognise its files          (`match`)
  2. how are its image streams laid out    (`driver` + `geometry`)
  3. where are its lenses pointing, and    (`lenses`)
     how far past 90 deg does each see
  4. where does its GPS/IMU live           (`telemetry`)

Everything downstream -- view planning, reframing, masking, ingest -- consumes
the profile through `splatpipe.drivers`, so adding a camera is a YAML file plus
(only if its projection is genuinely new) a driver. Profiles ship in
`splatpipe/data/profiles`; `SPLATPIPE_PROFILES=/path` or `--profile-dir` adds
more, and a user directory wins over a bundled file of the same name.

THE LENS LIST IS THE POINT. A 360 camera is not one camera: it is two (or
more) real cameras, centimetres apart, each seeing slightly past a hemisphere.
Consumer software hides that by warping the overlap until the parallax
disappears -- great for viewing, actively wrong for reconstruction, because a
warped overlap is two viewpoints' geometry averaged into one image. We keep
the lenses separate all the way to the trainer, so the baseline stays what it
physically is: extra information.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

BUNDLED = Path(__file__).parent / "data" / "profiles"


@dataclass(frozen=True)
class Lens:
    """One physical imager on the rig.

    `axis` is the optical axis in the rig frame (x right, y forward, z up),
    the same frame `reframe.view_dirs` uses, so a lens axis and a view yaw are
    directly comparable. `center_mm` is the entrance pupil offset from the rig
    origin -- documentation and rig-constraint solvers only; the pipeline never
    needs it, because SfM recovers the baseline from the images themselves.
    """

    name: str
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_fov_deg: float | None = None    # physical coverage, None = ask the driver
    usable_half_fov_deg: float | None = None  # trim of the soft/flary outer ring

    @property
    def axis_np(self) -> np.ndarray:
        a = np.asarray(self.axis, dtype=np.float64)
        return a / np.linalg.norm(a)

    @property
    def yaw_deg(self) -> float:
        a = self.axis_np
        return float(np.degrees(np.arctan2(a[0], a[1])))


@dataclass(frozen=True)
class CameraProfile:
    name: str
    driver: str
    lenses: tuple[Lens, ...]
    telemetry: str = "none"
    match: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    priority: int = 0        # higher wins when several profiles match
    status: str = "unknown"  # validated | derived | untested
    notes: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict, source: str = "") -> "CameraProfile":
        lenses = tuple(Lens(**l) for l in d.get("lenses", []))
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in d.items() if k in known and k not in ("lenses", "source")}
        kw.setdefault("name", Path(source).stem or "unnamed")
        return cls(lenses=lenses, source=source, **kw)

    def lens(self, name_or_index) -> Lens:
        if isinstance(name_or_index, int):
            return self.lenses[name_or_index]
        for l in self.lenses:
            if l.name == name_or_index:
                return l
        raise KeyError(f"{self.name}: no lens {name_or_index!r}")


def profile_dirs() -> list[Path]:
    dirs = [BUNDLED]
    env = os.environ.get("SPLATPIPE_PROFILES", "")
    dirs += [Path(p) for p in env.split(os.pathsep) if p]
    return dirs


@lru_cache(maxsize=1)
def _load_all() -> dict[str, CameraProfile]:
    out: dict[str, CameraProfile] = {}
    for d in profile_dirs():           # later dirs override earlier ones
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            data = yaml.safe_load(path.read_text()) or {}
            prof = CameraProfile.from_dict(data, source=str(path))
            out[prof.name] = prof
    return out


def all_profiles() -> dict[str, CameraProfile]:
    return dict(_load_all())


def get(name: str) -> CameraProfile:
    profiles = _load_all()
    if name not in profiles:
        raise SystemExit(f"[profiles] unknown profile {name!r}; "
                         f"have: {', '.join(sorted(profiles))}")
    return profiles[name]


def _streams(video: Path) -> list[tuple[int, int]]:
    import json
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=width,height", "-of", "json", str(video)],
        check=True, capture_output=True).stdout
    return [(s["width"], s["height"]) for s in json.loads(out).get("streams", [])]


def detect(video: Path, hint: str | None = None) -> CameraProfile:
    """Pick a profile for `video`; `hint` short-circuits the probe."""
    if hint:
        return get(hint)
    suffix = video.suffix.lower()
    sizes = _streams(video)
    best: tuple[int, CameraProfile] | None = None
    for prof in _load_all().values():
        m = prof.match or {}
        if "suffix" in m and suffix not in [s.lower() for s in m["suffix"]]:
            continue
        if "tracks" in m and len(sizes) != int(m["tracks"]):
            continue
        score = prof.priority
        if "track_size" in m:
            want = [tuple(t) for t in m["track_size"]]
            if not sizes or tuple(sizes[0]) not in want:
                continue
            score += 100        # an exact resolution match beats a generic rule
        if best is None or score > best[0]:
            best = (score, prof)
    if best is None:
        raise SystemExit(
            f"[profiles] no profile matches {video.name} "
            f"({suffix}, {len(sizes)} video track(s), {sizes[:1]}). "
            f"Write one in {BUNDLED} or point SPLATPIPE_PROFILES at yours.")
    prof = best[1]
    print(f"[profiles] {video.name}: {prof.name} ({prof.driver}, {prof.status})")
    return prof
