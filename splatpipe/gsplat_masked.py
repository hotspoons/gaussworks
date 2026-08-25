# SPDX-License-Identifier: Apache-2.0
"""Run gsplat's reference trainer with our per-camera masks applied.

gsplat's trainer already understands masks -- it zeroes masked pixels in the
render and the loss (`render_colors[~masks] = 0`) -- but its COLMAP parser
hardcodes `mask_dict[camera_id] = None`, so masks written for COLMAP never
reach training. Without this the model spends capacity fitting the capture
vehicle, which sits on the same pixels in every frame yet is somewhere
different in the world each time: it cannot converge to anything, and it
leaves a blurry blob along the whole driven path.

We patch `Parser.__init__` to populate `mask_dict` from `<chunk>/masks/<cam>/`
and then hand argv straight to `simple_trainer`, so we inherit its CLI and stay
out of its way.
"""

import runpy
import sys
from pathlib import Path

import cv2
import numpy as np


def _install_mask_patch(examples: Path):
    sys.path.insert(0, str(examples))
    import datasets.colmap as colmap_ds            # noqa: PLC0415

    original = colmap_ds.Parser.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        data_dir = Path(getattr(self, "data_dir", kwargs.get("data_dir", "")))
        masks_root = data_dir / "masks"
        if not masks_root.is_dir():
            return
        # camera_id -> camera folder, recovered from the image paths COLMAP
        # recorded (single_camera_per_folder means one id per camN directory)
        cam_folder = {}
        for cam_id, path in zip(self.camera_ids, self.image_paths):
            cam_folder.setdefault(int(cam_id), Path(path).parent.name)

        applied = 0
        for cam_id, folder in cam_folder.items():
            # chunks carry one symlink per image (COLMAP's convention) and not
            # necessarily the shared _mask.png, so fall back to any of them --
            # every mask for a camera is the same image by construction
            cam_masks = masks_root / folder
            shared = cam_masks / "_mask.png"
            if not shared.exists():
                candidates = sorted(cam_masks.glob("*.png"))
                if not candidates:
                    continue
                shared = candidates[0]
            mask = cv2.imread(str(shared), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            w, h = self.imsize_dict[cam_id]
            if (mask.shape[1], mask.shape[0]) != (w, h):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self.mask_dict[cam_id] = mask > 127     # True = keep
            applied += 1
        print(f"[masked] applied {applied}/{len(cam_folder)} camera masks "
              f"({100 * float((self.mask_dict[list(cam_folder)[0]] == 0).mean()):.0f}% "
              f"of frame masked)" if applied else "[masked] no masks found")

    colmap_ds.Parser.__init__ = patched


def main():
    examples = Path(sys.argv[1])
    _install_mask_patch(examples)
    sys.argv = [str(examples / "simple_trainer.py"), *sys.argv[2:]]
    runpy.run_path(str(examples / "simple_trainer.py"), run_name="__main__")


if __name__ == "__main__":
    main()
