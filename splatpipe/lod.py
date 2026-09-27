# SPDX-License-Identifier: Apache-2.0
"""Stage 5b: cheaper copies of a world's tiles -- a far LOD, and a probe LOD.

Driving shows the corridor AHEAD as well as the cell you are in, so the far
tiles need to exist without costing what the near ones do. Two independent
levers, measured on this capture's tiles:

  SPHERICAL HARMONICS are 73% of the bytes. A gaussian carries 45 f_rest
  coefficients (bands 1-3) against 14 other properties, so dropping them is a
  4x file reduction for a loss that is entirely view-dependent shading --
  specular variation on paint and wet road. At distance, and on a diffuse
  suburban scene, that is close to free. Measured: 97 MB -> 23 MB per tile.

  COUNT is what a rasteriser actually pays. Keep the gaussians that contribute
  most and drop the rest -- contribution ranked by opacity x footprint, since a
  large faint gaussian and a small bright one can matter equally and neither
  alone is the right key.

The two compose, and which one matters depends on who is asking. A browser on
a real GPU is bandwidth-bound and wants the SH gone; a software rasteriser in
a headless probe is count-bound and wants the decimation. `--keep` and
`--sh` are therefore separate.

This never touches the source tiles: LODs are written beside them, and
world.json gains a `lods` block naming what exists at each level.
"""

import json
import math
from pathlib import Path

import numpy as np

from . import ply


def _contribution(data: np.ndarray) -> np.ndarray:
    """Rank gaussians by how much of the image they can possibly affect.

    opacity is stored pre-sigmoid and scale pre-exp (the trainer's
    parameterisation), so both are mapped back before they mean anything --
    ranking on the raw stored values silently ranks on the wrong quantity.
    """
    op = data["opacity"].astype(np.float64)
    alpha = 1.0 / (1.0 + np.exp(-op))
    s = np.stack([data[f"scale_{i}"].astype(np.float64) for i in range(3)], axis=1)
    ex = np.exp(np.clip(s, -30, 20))
    # the two largest axes: a gaussian's silhouette is an ellipse, and the
    # third axis is its thickness, which costs nothing on screen
    ex.sort(axis=1)
    area = ex[:, 1] * ex[:, 2]
    return alpha * area


def decimate(data: np.ndarray, keep: float) -> np.ndarray:
    if keep >= 1.0:
        return data
    n = max(1, int(round(len(data) * keep)))
    c = _contribution(data)
    idx = np.argpartition(-c, n - 1)[:n]
    idx.sort()                      # keep file order stable and locality-friendly
    return data[idx]


def drop_sh(data: np.ndarray) -> np.ndarray:
    names = [n for n in data.dtype.names if not n.startswith("f_rest_")]
    if len(names) == len(data.dtype.names):
        return data
    out = np.zeros(len(data), dtype=np.dtype([(n, data.dtype[n]) for n in names]))
    for n in names:
        out[n] = data[n]
    return out


def build(world: Path, keep: float = 0.125, sh: bool = False,
          name: str = "far") -> Path:
    """Write one LOD level for every tile of a merged world."""
    wj = json.loads((world / "world.json").read_text())
    out = world / f"tiles_{name}"
    out.mkdir(parents=True, exist_ok=True)
    src_bytes = kept_bytes = 0
    src_n = kept_n = 0
    rows = []
    for t in wj["tiles"]:
        src = world / "tiles" / t["tile"]
        if not src.exists():
            continue
        data, _ = ply.read(src)
        src_bytes += src.stat().st_size
        src_n += len(data)
        d = decimate(data, keep)
        if not sh:
            d = drop_sh(d)
        dst = out / t["tile"]
        ply.write(dst, d)
        kept_bytes += dst.stat().st_size
        kept_n += len(d)
        rows.append({"tile": t["tile"], "gaussians": int(len(d))})
        print(f"[lod:{name}] {t['chunk']}: {len(data)} -> {len(d)} gaussians, "
              f"{src.stat().st_size/1e6:.1f} -> {dst.stat().st_size/1e6:.1f} MB", flush=True)
    wj.setdefault("lods", {})[name] = {
        "dir": out.name, "keep": keep, "sh": sh,
        "gaussians": kept_n, "tiles": rows,
    }
    (world / "world.json").write_text(json.dumps(wj, indent=1))
    print(f"[lod:{name}] {len(rows)} tiles, {src_n} -> {kept_n} gaussians "
          f"({100*kept_n/max(src_n,1):.1f}%), "
          f"{src_bytes/1e6:.0f} -> {kept_bytes/1e6:.0f} MB "
          f"({kept_bytes/max(src_bytes,1):.1%} of source) -> {out}")
    return out
