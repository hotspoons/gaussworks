#!/usr/bin/env python
"""Is a solved chunk geometrically sane? usage: pose-check.py CHUNK [SPARSE_SUBDIR=0]

Per corridor pass: SfM-vs-GPS residual, SfM height scatter (a road is flat to
well under a metre), nearest distance between passes (same road -> a lane
width), and focal drift from the exact synthetic intrinsics.
"""
import json, sys
from pathlib import Path
import numpy as np, pycolmap

chunk = Path(sys.argv[1]); sub = sys.argv[2] if len(sys.argv) > 2 else "0"
rec = pycolmap.Reconstruction(str(chunk / "sparse" / sub))
cams = {c["cam"]: c for c in json.load(open(chunk / "cameras.json"))}
folder = {}
for im in rec.images.values(): folder.setdefault(im.camera_id, im.name.split("/")[0])
print(f"{rec.num_reg_images()} registered, {rec.num_points3D()} points, mean reproj {rec.compute_mean_reprojection_error():.3f} px")
for cid, cam in rec.cameras.items():
    fx0 = cams[folder[cid]]["fx"]; print(f"  {folder[cid]}: fx {cam.params[0]:.1f} fy {cam.params[1]:.1f} (exact {fx0:.1f}, drift {100*(cam.params[0]/fx0-1):+.2f}%)")
ref = {}
for line in open(chunk / "geo_enu.txt"):
    n, x, y, z = line.split(); ref[n] = np.array([float(x), float(y), float(z)])
corp = chunk / "corridor_gps.json"
cor = json.load(open(corp if corp.exists() else chunk / "corridor.json"))
ranges = [tuple(p["seq_range"]) for p in cor["passes"]]
pass_of = lambda seq: next((i for i, (a, b) in enumerate(ranges) if a <= seq <= b), -1)
rows = []
for im in rec.images.values():
    if not im.has_pose or im.name not in ref: continue
    seq = int(Path(im.name).stem); c = im.projection_center(); g = ref[im.name]
    rows.append((pass_of(seq), seq, np.linalg.norm(c[:2] - g[:2]), c[2] - g[2], im.name.startswith("cam1/"), c))
tr = {}
for pi in range(len(ranges)):
    sel = [r for r in rows if r[0] == pi]
    if not sel: continue
    h = np.array([r[2] for r in sel]); v = np.array([r[3] for r in sel]); z = np.array([r[5][2] for r in sel])
    tr[pi] = np.array([r[5] for r in sel if r[4]])
    print(f"pass {pi} seq {ranges[pi]}: n={len(sel):4d}  GPS resid horiz med {np.median(h):5.1f} p90 {np.percentile(h,90):5.1f} m, vert mean {v.mean():+5.1f} m | SfM z mean {z.mean():6.1f} std {z.std():5.2f} m")
ks = sorted(tr)
for i, a in enumerate(ks):
    for b in ks[i+1:]:
        if len(tr[a]) and len(tr[b]):
            d = np.array([np.min(np.linalg.norm(tr[b][:, :2] - q[:2], axis=1)) for q in tr[a]])
            near = d < 25   # only where the passes actually overlap
            print(f"pass {a} vs {b}: nearest horiz median {np.median(d):5.1f} m; where overlapping (<25 m, {near.mean():.0%} of pass {a}): median {np.median(d[near]) if near.any() else float('nan'):4.1f} m, dz {abs(tr[a][:,2].mean()-tr[b][:,2].mean()):.1f} m")
