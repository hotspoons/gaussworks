# SPDX-License-Identifier: Apache-2.0
"""Compare trained models fairly, which the trainer's own PSNR cannot.

gsplat zeroes masked pixels in the render but not in the ground truth, so a
mask-trained model is scored against the vehicle it was told to ignore: on our
capture that is 36% of every frame counted as pure error, and reported PSNR
drops ~4.5 dB for a model that may in fact be better. Comparing runs by that
number silently rewards NOT masking.

This scores only the pixels a model was actually asked to reproduce -- the
unmasked region -- so masked and unmasked runs are directly comparable, and
so are two masked runs with different mask coverage.
"""

import sys
from pathlib import Path

import numpy as np


def _val_indices(names: list[str], test_every: int = 8,
                 only: set[str] | None = None) -> list[int]:
    """Held out by frame number (holdout.py), so the same physical frames are
    scored whichever image subset a chunk was cut with. `only` narrows that to
    a shared list -- the intersection of two variants' held-out frames is the
    only set on which their PSNRs can be compared."""
    from .holdout import is_holdout                # noqa: PLC0415
    idx = [i for i, n in enumerate(names) if is_holdout(n, i, test_every)]
    if only is not None:
        idx = [i for i in idx if names[i] in only]
    return idx


def evaluate(chunk: Path, ckpts: list[Path], examples: Path | None = None,
             test_every: int = 8, at_width: int | None = None,
             names: Path | None = None) -> dict:
    import cv2                                     # noqa: PLC0415
    import torch                                   # noqa: PLC0415
    from gsplat import rasterization               # noqa: PLC0415

    from .mesh import _load_parser, _load_splats   # noqa: PLC0415

    chunk = Path(chunk)
    parser = _load_parser(chunk)
    only = None
    if names is not None:
        only = {ln.strip() for ln in Path(names).read_text().splitlines() if ln.strip()}
    idx = _val_indices(list(parser.image_names), test_every, only)
    if not idx:
        raise SystemExit(f"[eval] {chunk.name}: no held-out views"
                         + (f" in common with {names}" if names else ""))
    print(f"[eval] {chunk.name}: {len(idx)} held-out views"
          + (f" (shared list {Path(names).name})" if names else ""), flush=True)

    masks = {}
    for cam_id, path in zip(parser.camera_ids, parser.image_paths):
        cam_id = int(cam_id)
        if cam_id not in masks:
            found = sorted((chunk / "masks" / Path(path).parent.name).glob("*.png"))
            m = cv2.imread(str(found[0]), cv2.IMREAD_GRAYSCALE) if found else None
            masks[cam_id] = (m > 127) if m is not None else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for ckpt in ckpts:
        splats = _load_splats(Path(ckpt), device)
        sh_degree = int(round(splats["sh"].shape[1] ** 0.5)) - 1
        psnrs = []
        for i in idx:
            cam_id = int(parser.camera_ids[i])
            K = torch.tensor(parser.Ks_dict[cam_id], dtype=torch.float32, device=device)
            c2w = torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device=device)
            gt = cv2.cvtColor(cv2.imread(parser.image_paths[i]),
                              cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h, w = gt.shape[:2]
            keep = masks.get(cam_id)
            if at_width and at_width != w:
                # Render and score at a COMMON width. Two models trained at different
                # view resolutions cannot be compared at their own: the higher-resolution
                # one is scored against ground truth carrying far more high-frequency
                # detail, so equal PSNR means a harder target met, not equal quality --
                # and the number says nothing about which world looks better.
                # INTER_AREA because this is a downscale; anything else aliases the
                # ground truth and scores the model against the aliasing.
                sc = at_width / w
                nh = max(1, int(round(h * sc)))
                gt = cv2.resize(gt, (at_width, nh), interpolation=cv2.INTER_AREA)
                if keep is not None:
                    keep = cv2.resize(keep.astype(np.uint8), (at_width, nh),
                                      interpolation=cv2.INTER_NEAREST) > 0
                K = K.clone()
                K[0, :] *= sc
                K[1, :] *= sc
                h, w = nh, at_width
            with torch.no_grad():
                img, _, _ = rasterization(
                    splats["means"], splats["quats"], splats["scales"],
                    splats["opacities"], splats["sh"],
                    torch.linalg.inv(c2w)[None], K[None], w, h,
                    sh_degree=sh_degree, render_mode="RGB")
            pred = img[0].clamp(0, 1).cpu().numpy()
            diff = (pred - gt) if keep is None else (pred - gt)[keep]
            psnrs.append(10 * np.log10(1.0 / max(float((diff ** 2).mean()), 1e-12)))
        results[str(ckpt)] = {"psnr_visible": round(float(np.mean(psnrs)), 3),
                              "views": len(idx), "at_width": at_width,
                              "gaussians": int(len(splats["means"]))}
        at = f"@{at_width}px  " if at_width else ""
        print(f"[eval] {Path(ckpt).parent.parent.name:22s} "
              f"PSNR(visible) {results[str(ckpt)]['psnr_visible']:6.2f} dB  {at}"
              f"{results[str(ckpt)]['gaussians']:>8,} gaussians", flush=True)
    return results
