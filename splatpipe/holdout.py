# SPDX-License-Identifier: Apache-2.0
"""Which views are held out of training, decided by FRAME, not by list index.

gsplat's reference split is `index % test_every == 0` over the sorted image
list. Two things are wrong with that for a rig capture, and both were found
while trying to compare camera-selection rules:

1. The held-out set depends on which images the chunk happens to contain.
   Split the same cell three ways (crop / cell / inria) and each rule holds
   out DIFFERENT views, so their PSNRs are not comparable -- and the rules
   that keep only local cameras hold out easier, more local views.
2. A rig writes six views per frame (cam0..cam5, same position, same
   instant). Holding out cam0/000874 while training on cam1..cam5 of the same
   frame is a leak: the model has seen that spot from that moment, just a few
   degrees rotated. Every PSNR so far was optimistic by an unknown amount.

Holding out by frame NUMBER fixes both: frame 000874 is held out in every
variant that contains it, all six of its views together. Comparing two
variants then means scoring them on the frames they BOTH hold out.
"""

import re
from pathlib import Path

_DIGITS = re.compile(r"(\d+)(?!.*\d)")


def frame_number(name: str) -> int | None:
    """000874 for cam3/000874.jpg; None when a name carries no number."""
    m = _DIGITS.search(Path(name).stem)
    return int(m.group(1)) if m else None


def is_holdout(name: str, index: int, every: int = 8) -> bool:
    """One frame in `every` is held out, chosen by its number so that the
    decision does not move when the image list changes. Falls back to the
    index rule for names without a frame number."""
    if every <= 0:
        return False
    n = frame_number(name)
    return (index if n is None else n) % every == 0


def holdout_names(names, every: int = 8) -> list[str]:
    return [n for i, n in enumerate(names) if is_holdout(n, i, every)]


def common_holdout(*name_lists, every: int = 8) -> list[str]:
    """The held-out frames every variant shares -- the only set on which two
    variants' PSNRs mean the same thing."""
    sets = [set(holdout_names(ns, every)) for ns in name_lists]
    return sorted(set.intersection(*sets)) if sets else []
