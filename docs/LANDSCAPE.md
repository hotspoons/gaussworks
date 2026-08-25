<!-- SPDX-License-Identifier: Apache-2.0 -->
# Tools, pitfalls, and licensing

Running notes for gaussworks. Three questions this answers: what could we use,
what has already bitten us, and what may we legally ship.

gaussworks is **Apache-2.0**, so anything we *bundle, link, or derive from*
must be permissive (MIT / BSD / Apache). Copyleft and research-only components
can still be used as **separate tools a user chooses to run** — that is a very
different thing from vendoring them, and the distinction is what most of the
licence column below is about.

## Licence rules of thumb

| Tier | Licences | What we may do |
| --- | --- | --- |
| Safe to depend on | MIT, BSD-2/3, Apache-2.0 | vendor, link, derive, ship |
| Invoke only | GPL-3.0, AGPL-3.0 | user installs it and runs it themselves; never bundled, never linked, no derived code |
| Never | research/non-commercial (INRIA-style) | not even for outputs we intend to ship |
| Data | CC BY (attribute), CC BY-SA (viral on derivatives) | see THIRD_PARTY.md |

## In use today

| Component | Licence | Role | Notes |
| --- | --- | --- | --- |
| [COLMAP](https://github.com/colmap/colmap) | BSD-3 | SfM / poses | binaries link LGPL deps from apt — fine internally, revisit before shipping an image publicly |
| [GLOMAP](https://github.com/colmap/glomap) | BSD-3 | global SfM | faster mapper, drop-in |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | splat training | chosen over INRIA 3DGS specifically for the licence |
| [Open3D](https://www.open3d.org/) | MIT | TSDF fusion, mesh IO | mesh stage |
| ffmpeg / ExifTool | LGPL-GPL / Artistic-GPL | decode, GPMF telemetry | invoked as subprocesses, never linked |
| numpy, OpenCV, PyYAML, requests, piexif | BSD / Apache / MIT | pipeline | |

## Candidates (not yet used)

| Tool | Licence | Why we care | Verdict |
| --- | --- | --- | --- |
| [Brush](https://github.com/ArthurBrussee/brush) | **Apache-2.0** | splatting in Rust on **wgpu — no CUDA**, so AMD/Apple/laptop GPUs can train | **strongest lead for the resource-floor contract**; evaluate as an alternative trainer |
| [mvs-texturing](https://github.com/nmoehrle/mvs-texturing) | check before use (GitHub reports no standard licence) | UV atlas + photo texture bake | the missing piece between our vertex-coloured mesh and a sim-ready asset; **confirm licence first** |
| [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) | MIT | surface from oriented points | cleaner alternative to TSDF for the mesh stage |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | **AGPL-3.0** | dense MVS | invoke-only. Do **not** bundle. Earlier notes suggested it too casually |
| RealityCapture | proprietary, free under revenue threshold | best-in-class photogrammetry mesh | user-run; good AC/UE path, ingests our poses |
| [SuGaR](https://github.com/Anttwo/SuGaR), [2DGS](https://github.com/hbb1/2d-gaussian-splatting) | INRIA research-only | splat→mesh | **excluded**, hence our own depth-fusion route |
| [simple_photogrammetry_gui](https://github.com/edin45/simple_photogrammetry_gui) | GPL-3.0 | Flutter GUI wrapping the above | good ideas + parts list; no code borrowing |
| [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) | MIT | splats in Unity | the hybrid engine path |

## 360 stitching: surveyed, then deliberately not used

Full write-up in [SEAM.md](SEAM.md). Short version: every open stitcher solves
a problem we do not have, and solving it damages the input we do need.

| Tool | Licence | What it does with the lens overlap | Verdict |
| --- | --- | --- | --- |
| [max2sphere](https://github.com/trek-view/max2sphere) | Apache-2.0 | hard pick per pixel | **in use** — our EAC decoder derives from it (see THIRD_PARTY.md) |
| ffmpeg [`vf_gopromax_opencl`](https://patchwork.ffmpeg.org/project/ffmpeg/patch/20240803005601.44246-2-aimingoff@pc.nifty.jp/) | LGPL | linear alpha ramp over the 64 px strip | reference implementation of .360 → equirect; **ghosts near objects exactly as our old cross-fade did** |
| GoPro D.WARP (in-camera / Player) | proprietary, patented ([11568516](https://patents.google.com/patent/US11568516), [11748952](https://patents.google.com/patent/US11748952)) | depth/optical-flow local warp until disparity cancels | best-looking, **worst input** — destroys the disparity and breaks the central-projection assumption SfM needs |
| [Surround360](https://github.com/facebookarchive/Surround360) / [Panorama-OpticalFlow](https://github.com/MungoMeng/Panorama-OpticalFlow) | BSD / MIT | optical-flow stitch | same objection as D.WARP |
| Hugin / enblend | GPL | seam finding + multiband blend | invoke-only anyway; no parallax model |
| [Seam360GS](https://arxiv.org/abs/2508.20080) (ICCV 2025) | paper, CC-BY | models the two optical centres explicitly instead of stitching | **agrees with our approach**; worth reading if we ever want a rig-constrained trainer |

Our answer is `splatpipe/viewplan.py`: plan the virtual cameras *inside* each
lens' cone so no training image ever contains a join. Same image count as the
old ring, no seam to blend.

## Pitfalls (all hit for real, all cost hours)

**Build / environment**

- **The NVIDIA PyTorch containers export `TORCH_CUDA_ARCH_LIST` with every arch** (`7.5 8.0 8.6 9.0 10.0 12.0+PTX`). gsplat's JIT then builds seven device variants of every kernel; the fused rasterizers took >1 h each and OOM-killed a 32 GB box three times. Override from `nvidia-smi --query-gpu=compute_cap`. See `scripts/gsplat-env.sh`.
- **gsplat JIT-compiles on first import**, into `TORCH_EXTENSIONS_DIR` (default `~/.cache` = container-fs = recompiled after every restart). Put it on the PVC.
- **torch hashes the build config into the cache.** Launch the trainer and the viewer with different flags and ninja rebuilds everything. One sourced env file for all entry points.
- **Container filesystem is ephemeral; only the PVC survives.** A PVC-built COLMAP still needs its apt-installed shared libraries reinstalled after every restart (`libOpenGL`, `libGLEW`, `libceres`, boost, metis, freeimage…) or it exits 127. That is what `scripts/pod-bootstrap.sh` is for.
- **Unbounded compile concurrency kills the node, not just the pod.** A single `cicc` peaks 9–20 GB. Size jobs off free RAM, and set a container memory limit so the kubelet survives.
- **`pkill -f <pattern>` inside `kubectl exec` matches the exec's own command line** and kills your shell — it looks like a mystery exit 137/143. Hit three times. Kill by exact name (`pkill -x cicc`).
- Python buffers stdout under `nohup`: a log that stays 0 bytes for 30 minutes is usually just buffering. Use `flush=True`.

**Reconstruction**

- **Tell COLMAP the intrinsics.** We synthesise pinhole views, so fx/fy/cx/cy are exact — but left to guess COLMAP assumes `fx = 1.2·max(w,h)`, which for a 100° view is 1920 against a true 671. The mapper cannot bootstrap: **5/424 images registered, versus 385/424 once told.** Biggest single bug so far.
- **`model_aligner --alignment_type enu` centres each chunk on its own GPS centroid**, so chunks never share a frame. Align to project-frame positions with `custom` instead; then merging is concatenation and needs no SH rotation.
- **Sequential matching never crosses camera folders** on a multi-view rig. Use spatial (GPS priors) or exhaustive on small chunks.
- **Chunk by locality, not by distance along the track**, or the second pass down a road competes with the first instead of reinforcing it.
- **`sorted()` on checkpoints is lexicographic**: `ckpt_14999` sorts before `ckpt_6999`, so `[-1]` silently picks the earlier model. Sort by parsed step. Bug appeared twice.
- **Rendered "expected depth" is not a depth map.** It interpolates across object silhouettes and returns plausible values for empty sky, which extrudes radial spikes and inflates a TSDF across the far field. Mask by alpha coverage and reject steep depth gradients.
- **Depth fusion needs real parallax.** A short handheld clip produces geometry-shaped noise no filtering can rescue.

**Capture**

- **The capture vehicle is rigid in the camera frame**, so it lands on the same pixels forever: COLMAP matches features on your own roof, and the trainer fits a surface that is somewhere different in every frame. One static mask per camera fixes both.
- **Staticness and edge tests fail on glossy paint/glass** — moving reflections make a roof look dynamic. Keying on the temporal median being darker works; hand-drawn masks are the escape hatch.
- **gsplat supports per-camera masks but its COLMAP parser never loads them** (`mask_dict[camera_id] = None`). Masks reach COLMAP and not training unless you patch it -- see `splatpipe/gsplat_masked.py`.
- **A module named `queue.py` inside the package shadows the stdlib `queue`** for any script run from that directory; torch imports `from queue import Queue` deep in its stack and training dies. Renamed to `workqueue.py`.
- **gsplat normalises world space by default**, recentring and rescaling into a unit box -- which silently discards the shared ENU frame that merge, mesh and drive all depend on. Train with `--no-normalize-world-space`.
- **Virtual views can under-sample the sphere.** 4 × 100° at 1600 px is 16 px/deg, while 8K EAC faces hold ~21. Fixed: profiles carry a `px_per_deg` default (24 for Max 2) and the plan sizes views from it.
- **A view that straddles the lens boundary is two viewpoints in one image.** The old fixed yaw ring put the boundary inside four of six views, at yaw ±90° — broadside, where the nearest objects are. Cross-fading it grew duplicate geometry (a truck appearing twice, a roof overlapping itself); hard-cutting it leaves a discontinuity. Neither is fixable downstream. Plan views inside each lens' cone instead — [SEAM.md](SEAM.md).
- **Do not hardcode a camera's geometry into pipeline code.** The EAC template, lens count, lens axes, telemetry source and sensible view density are all *hardware facts*, and burying them in ingest made the Max 2 look like a special case and an Insta360 look like a rewrite. They now live in `splatpipe/data/profiles/*.yaml`, behind a driver interface that is three methods wide.
- Reframe at high shutter speed; motion blur is unfixable downstream.

## Measurements

Controlled runs on one 150 m chunk of real neighbourhood capture (942 images,
6 virtual views, 30k steps), scored with `splatpipe eval` on visible pixels
only -- the trainer's own PSNR is unusable here, see below.

| Config | PSNR (visible) | Gaussians |
| --- | --- | --- |
| no mask, no AA, no regularisers | 22.80 dB | 278,025 |
| mask + AA + opacity/scale reg | 21.87 dB | 171,280 |
| **mask + AA, no regularisers** | **23.07 dB** | **293,014** |

Conclusions: masking the capture vehicle plus anti-aliasing is the best
configuration; `--opacity-reg 0.001 --scale-reg 0.01` cost 1.2 dB and pruned
40% of the gaussians, so they are off by default. Applying them was a guess
(aimed at needle artefacts) bundled into a run with two other changes, which
is how a regression hides.

Separately, the ingest profile change alone (4 views x 100 deg at 1600 px ->
6 x 80 deg at 1920 px, spacing 1.75 -> 1.25 m) took an unmasked 15k-step
model from 18.93 to 20.32 dB all-pixel PSNR, and registration from 91% to
99.8% (385/424 -> 940/942).

**Never compare runs using the trainer's PSNR when masks differ.** gsplat
zeroes masked pixels in the render but not the ground truth, so a masked model
is scored against the vehicle it was told to ignore: 36% of frame here, ~4.5
dB of penalty. That metric would have concluded "never mask", exactly
backwards.

## Open questions
- Is Brush a viable second trainer, and does it unlock non-CUDA hardware?
- Which texture-bake path is both good and permissive (mvs-texturing licence pending)?
- Does depth fusion behave on driving capture, or do we switch to MVS?
- How much does per-lens view planning actually buy, in dB and in visible ghosting? (baseline to beat: **23.07 dB** on the street chunk, seam-blended ingest)
- Do the shipped `insta360-x3` / `insta360-x4` profiles survive contact with a real .insv? Their `geometry.circles` are from published specs, not measured.
