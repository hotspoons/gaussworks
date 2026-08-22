# Third-party components, data, and attributions

## Publishing plan and open concerns

Intent: publish the pipeline code under **Apache-2.0**, and eventually publish
the Docker image. The **code repo is Apache-clean today** (everything we
ported is Apache-2.0; our files carry SPDX headers; LICENSE included). The
**image is an aggregate work** and can never be "all Apache" — it must ship
with this notice file. Concerns to address before the image goes public:

1. **NVIDIA base image (blocker)**: `ai-dev-pod` builds on
   `nvcr.io/nvidia/pytorch`, distributed under the NVIDIA Deep Learning
   Container License — public redistribution needs a license review. Likely
   fix for a public image: rebase on `nvidia/cuda` runtime or Ubuntu + pip
   torch (note: torch wheels still bundle NVIDIA-EULA CUDA libs — universal
   practice, but the image still isn't purely open licenses; document it).
2. **ffmpeg (Ubuntu build)** links GPL components (x264). Distributing it in
   an image is fine with source availability (apt), but for a cleaner story
   build LGPL-only ffmpeg or make ffmpeg a documented host dependency.
3. **COLMAP binaries** link SuiteSparse, whose CHOLMOD/SPQR modules are GPL.
   BSD COLMAP source, GPL-linked binary: acceptable in the aggregate image
   with notices; the alternative is building Ceres/COLMAP without SuiteSparse
   (slower BA) if a GPL-free image is ever required.
4. **exiftool** (Artistic/GPL): subprocess-only; same aggregate reasoning as
   ffmpeg.
5. **Never bake data into the image or repo**: the CC BY test capture, the
   INRIA toy dataset (research-only!), and Mapillary pulls (CC BY-SA) are all
   fetch-on-demand into gitignored `data/`. Keep it that way.

## Code we build on or ported

| Component | Use | License | Notes |
| --- | --- | --- | --- |
| [trek-view/max2sphere](https://github.com/trek-view/max2sphere) | GoPro EAC face layout, split-half/blend arithmetic ported into `splatpipe/eac.py` | Apache-2.0 | Trek View's batch converter, based on Paul Bourke's max2sphere. Our implementation is a numpy re-derivation (per-face basis table); layout constants and split math come from this reference. |
| [COLMAP](https://github.com/colmap/colmap) | poses stage (features, matching, mapper, model_aligner); built into the Docker image | BSD-3-Clause (new BSD) | COLMAP itself is BSD; binaries link separately-licensed deps (e.g. LGPL SuiteSparse from Ubuntu) — fine for our internal use, revisit only if we ever distribute the image publicly. |
| [GLOMAP](https://github.com/colmap/glomap) | global SfM mapper (preferred over incremental); Docker image | BSD-3-Clause | |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | train stage (rasterizer + reference trainer); Docker image | Apache-2.0 | Deliberately chosen over INRIA 3DGS/H3DGS code, which is research-only (see below). |
| ffmpeg, ExifTool | invoked as subprocesses (frame extraction, GPMF telemetry) | LGPL/GPL; Artistic/GPL | Used as external CLI tools, not linked. |
| numpy (BSD-3), opencv-python-headless (Apache-2.0), pyyaml (MIT), requests (Apache-2.0), piexif (MIT) | python deps | permissive | |

## Test data

| Data | Source | License | Attribution |
| --- | --- | --- | --- |
| `GS010513.360` (`just fetch-360`) | [Zenodo record 21611765](https://zenodo.org/records/21611765), doi:10.5281/zenodo.21611765 | **CC BY 4.0** | Guo, J., Riaz, M., & Jensenius, A. R. — *360-Degree Camera Comparison Dataset* (AMBIENT project, RITMO, University of Oslo). Keep this attribution wherever the file or derived frames are used. |
| gpmf-parser samples (`just fetch-sample-360`) | [gopro/gpmf-parser](https://github.com/gopro/gpmf-parser) | Apache-2.0 | GoPro's telemetry parser repo; we use only its sample media. |
| H3DGS toy dataset (`just fetch-toy`) | [INRIA hierarchical-3d-gaussians](https://github.com/graphdeco-inria/hierarchical-3d-gaussians) | **INRIA research/evaluation-only, non-commercial — NOT permissive** | Optional benchmark only. Nothing in the pipeline depends on it; never derive shipped assets from it. |
| Mapillary imagery (`splatpipe mapillary`) | [Mapillary](https://www.mapillary.com) Graph API | **CC BY-SA 4.0** + Mapillary ToS | Fine for pipeline development/testing. ShareAlike applies to derivatives: do not build distributable game assets from Mapillary-sourced splats — own-capture footage is the asset path. |

## Own capture

Footage from our GoPro Max 2 / Hero rig is ours; splats derived from it carry
no third-party terms.
