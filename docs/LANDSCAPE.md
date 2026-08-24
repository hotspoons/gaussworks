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
- **gsplat supports per-camera masks but its COLMAP parser never loads them** (`mask_dict[camera_id] = None`). Masks reach COLMAP and not training unless you patch it. *(open)*
- **Virtual views can under-sample the sphere.** 4 × 100° at 1600 px is 16 px/deg, while 8K EAC faces hold ~21. *(open)*
- Reframe at high shutter speed; motion blur is unfixable downstream.

## Open questions

- Does masking in training measurably improve PSNR on foliage-heavy capture?
- Is Brush a viable second trainer, and does it unlock non-CUDA hardware?
- Which texture-bake path is both good and permissive (mvs-texturing licence pending)?
- Does depth fusion behave on driving capture, or do we switch to MVS?
