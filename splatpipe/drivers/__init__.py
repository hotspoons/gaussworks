# SPDX-License-Identifier: Apache-2.0
"""Driver registry. `driver:` in a profile is a key in here."""

from ..profiles import CameraProfile
from .base import Driver
from .dual_fisheye import DualFisheyeDriver
from .equirect import EquirectDriver
from .flat import FlatDriver
from .gopro_eac import GoProEacDriver

REGISTRY: dict[str, type[Driver]] = {
    "gopro_eac": GoProEacDriver,
    "dual_fisheye": DualFisheyeDriver,
    "equirect": EquirectDriver,
    "flat": FlatDriver,
}


def get_driver(profile: CameraProfile) -> Driver:
    try:
        cls = REGISTRY[profile.driver]
    except KeyError:
        raise SystemExit(f"[drivers] profile {profile.name!r} wants driver "
                         f"{profile.driver!r}; have: {', '.join(sorted(REGISTRY))}")
    return cls(profile)


__all__ = ["Driver", "get_driver", "REGISTRY"]
