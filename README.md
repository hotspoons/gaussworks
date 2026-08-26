# gaussworks — car-mounted 360 capture → gaussian splat pipeline

Turns wide-angle / 360 driving (and hiking) footage into chunked, geo-aligned
3D Gaussian Splat reconstructions. First target: the back-road network between
Crofton and Annapolis (see `configs/md-backroads.yaml`).

Canonical repo: https://github.com/hotspoons/gaussworks (mirrored on internal GitLab)
(worked on locally as a clone under `trailworks/ext/`). Self-contained: own
justfile, package, and container; Apache-2.0 (see LICENSE and THIRD_PARTY.md).
Outputs are geo-aligned (ENU), so they can eventually feed trailworks as a
detail layer.

## Two targets

**Drivable stages.** Car-mounted 360 capture of Maryland back roads → a
geo-aligned splat world → mesh/export for a moddable racing sim (Assetto Corsa
first). Everything measured in this repo so far comes from this path.

**Trail previews for trailworks.** The same pipeline, walked or ridden
instead of driven: a 360 camera on a backpack pole or a bike, and a railed browser flythrough so
someone can preview a trail before driving to the trailhead. The corridor
machinery already serves it — `guardrail.py` clamps a camera to the observed
path, and for a trail *the rail is the trail*. Not yet attempted; the capture
deltas, the canopy-GPS risk, the three.js renderer options and the open UX
question are worked out in [docs/TRAILVIEW.md](docs/TRAILVIEW.md), with a
starting configs in `configs/trail-hike.yaml` and `configs/trail-bike.yaml`.

## Pipeline

```
any camera with a profile (.360, .insv,        Mapillary sequences (equirect + GPS)
 equirect .mp4, plain pinhole)                            │
        │                                                 ▼
        ▼
  [1] ingest ──────────────────────────► frames.jsonl + images/camN/*.jpg + geo.txt
        profile picks a driver, driver maps rays → pixels, view plan picks the
        rays. GPS via exiftool, distance-spaced frame pick (sharpest per
        window), EXIF GPS written into each jpg. Every view is rendered
        THROUGH ONE LENS — see "The lens seam" below.
        │
        ▼
  [2] chunk        locality grid: ~200m cells + halo. Every pass through a
        │            cell feeds that cell's chunk, so driving a road twice
        │            strengthens one reconstruction instead of making two.
        │            Also emits corridor.json (observed envelope) per chunk.
        ▼
  [3] poses        per chunk: COLMAP features → spatial matching (GPS priors)
        │            → mapper (GLOMAP if present) → model_aligner to ENU
        ▼
  [4] train        per chunk: gsplat trainer, one GPU per chunk
        │
        ▼
  [5] merge        chunks -> world: each cell contributes only the gaussians
        │            it owns (halo donates observations, not geometry), floaters
        ▼            outside the capture corridor are pruned
     world/tiles/chunk_xN_yN.ply + world.json   (LOD hierarchy: next milestone)
```

Stages 3 and 4 pull work from a claim-based queue on shared storage, so any
number of workers on any number of nodes can be pointed at the same chunk
directory: work self-balances, a dead worker's chunk is reclaimed, and a
re-run is a no-op for anything already done.

## Cameras

Nothing hardware-specific lives in the pipeline. A **profile** is a YAML file
naming a **driver** (a projection), the lens axes, the telemetry source, and
sensible defaults:

```
$ splatpipe profiles
equirect-360           equirect       validated  telemetry=exif   lenses: sphere@+0
gopro-360-generic      gopro_eac      derived    telemetry=gpmf   lenses: front@+0, rear@+180
gopro-max              gopro_eac      validated  telemetry=gpmf   lenses: front@+0, rear@+180
gopro-max2             gopro_eac      validated  telemetry=gpmf   lenses: front@+0, rear@+180
insta360-x3            dual_fisheye   untested   telemetry=exif   lenses: front@+0, rear@+180
insta360-x4            dual_fisheye   untested   telemetry=exif   lenses: front@+0, rear@+180
pinhole                flat           validated  telemetry=exif   lenses: main@+0
```

Detection is automatic from the file (`splatpipe profiles clip.360 --plan`);
`--profile NAME` pins it. **Adding a camera whose projection we already speak
is a YAML file and no code.** Adding a new projection is one `Driver` subclass
with three methods. `SPLATPIPE_PROFILES=/path` loads yours without forking.

`status:` is not decoration — `untested` means the geometry came from a spec
sheet, and `splatpipe verify` exists to fix that against real footage.

### The lens seam

A two-lens 360 camera is **two cameras a few centimetres apart**, and the
directions where their coverage meets exist twice, from two different places.
Every consumer stitcher hides that by warping the overlap until the parallax
cancels. That is right for viewing and wrong for us: parallax is disparity,
disparity is depth, and a warped image is no longer a central projection —
which is the one thing structure-from-motion assumes it has.

So gaussworks does not stitch. It **plans the virtual cameras inside each
lens' cone**, and no training image ever contains a join:

```
[eac] 5952x1920: side=2016 face=1920 blend=96px -> each lens sees 94.74 deg from its axis
[viewplan] 6 view(s) / frame
   front: 3 x 80 deg 1920x1440 (24.0 px/deg)  yaw -50.0, 0, 50.0
    rear: 3 x 80 deg 1920x1440 (24.0 px/deg)  yaw 130.0, 180, 230.0
  covers 99.2% of the horizon (+-10 deg band), 34.2% of the full sphere
```

Same six images per frame the old fixed yaw ring produced, none of them
stitched — and where the lenses do overlap the pipeline now gets two views
with a real baseline instead of one image with a contradiction. Full
reasoning, prior art, and what GoPro's D.WARP actually does:
[docs/SEAM.md](docs/SEAM.md).

\* `.360` (GoPro EAC, two-track) ingest is native: `splatpipe/eac.py` remaps
EAC -> pinhole in one resample (no equirect intermediate, no patched ffmpeg),
validated on real Max and Max 2 footage. **Requires exiftool 12.90+** — the
Max 2 writes GPS9 telemetry, and older builds silently report no GPS at all.

## Layout

```
splatpipe/        python package (splatpipe CLI)
configs/          capture campaigns + stage parameters
deploy/           ZipspaceDeployment manifest for the cluster
data/             local working data (gitignored)
Dockerfile        FROM ai-dev-pod, adds COLMAP+GLOMAP (CUDA), gsplat, ffmpeg, exiftool
```

## Quickstart (local dev box)

```bash
just setup                 # venv + editable install
just fetch-360             # real raw GoPro Max .360 (3.8GB, CC BY 4.0)
just smoke                 # .360 → ingest → chunk → poses (train needs a GPU + gsplat)
just fetch-sample-360      # optional: small GoPro samples w/ GPMF telemetry
```

Real 360 data before the camera arrives, two ways:

```bash
# Mapillary: car-mounted 360 sequences w/ GPS, original resolution.
# Token: mapillary.com/dashboard/developers (free). Bbox is w,s,e,n.
export MAPILLARY_TOKEN=MLY...
just mapillary "-76.62,38.95,-76.55,39.02" data/mapillary-run

# YouTube 360 (heavier compression; plumbing tests only)
just fetch-yt "https://www.youtube.com/watch?v=..." data/yt-run
```

Then:

```bash
.venv/bin/splatpipe verify data/yt-run/video.mp4     # check the profile first
.venv/bin/splatpipe ingest data/yt-run/video.mp4 --out data/run1
.venv/bin/splatpipe chunk  --frames data/run1
.venv/bin/splatpipe poses  --chunks data/run1/chunks
.venv/bin/splatpipe train  --chunks data/run1/chunks
```

> **Platform reinstall / new pod onto the existing volume?** [docs/WORKSPACE-RESTORE.md](docs/WORKSPACE-RESTORE.md).
>
> **Starting from a fresh pod or handing this to someone else?**
> [docs/HANDOFF.md](docs/HANDOFF.md) is the executable version of everything
> below: provisioning from a raw `/workspace`, six verification gates, what to
> copy up, current state of the work, and the traps that cost hours.

## Remote dev pod (Zipspace, single GPU)

`deploy/devpod.yaml` stands up a 1-GPU remote-dev pod (VS Code plugin
connectable; `zip-friends` init provides the tunnel bins). The loop:

```bash
kubectl apply -f deploy/devpod.yaml
kubectl exec -it -n default <pod> -c dev -- bash -l
# first time on a fresh PVC:
sudo chown 1000:1000 /workspace
git clone https://github.com/hotspoons/gaussworks.git /workspace/gaussworks
cd /workspace/gaussworks && pip install -e . && export PATH=$HOME/.local/bin:$PATH
bash scripts/pod-bootstrap.sh    # ephemeral bits; scripts/pod-build-stack.sh for full stack
# iterate: edit anywhere, push, then here:
git pull --ff-only
```

## On the cluster (Zipspace)

Build/push the image, deploy `deploy/zipspace.yaml`, connect to the leader:

```bash
just image-build && just image-push
kubectl apply -f deploy/zipspace.yaml
```

Everything under `/workspace` is the shared RWX PVC, which is what lets the
queue coordinate workers. On the leader:

```bash
devpod launch python -m splatpipe.cli poses --chunks /workspace/data/run1/chunks
devpod launch python -m splatpipe.cli train --chunks /workspace/data/run1/chunks
splatpipe status --chunks /workspace/data/run1/chunks     # audit any time
```

Every worker loops "claim an unclaimed chunk, do it, mark it done" until the
pool is empty, pinned to `LOCAL_RANK`'s GPU — so a 4-node × 4-GPU group works
16 chunks at a time and rebalances itself when one chunk runs long. Failures
are retried (twice by default) and then recorded, so `status` tells you which
chunks need attention instead of the run dying. Scale is set by the group
size, not the code: a 22 km neighbourhood capture is ~140 chunks ≈ 4 h of
training on 16 GPUs, or a long afternoon on one.

Hardware mapping (see the fleet):

| Stage | Pool |
| --- | --- |
| ingest (decode-bound) | L40S |
| poses (SIFT GPU + CPU BA) | L40S / GH200 (big Grace CPUs for GLOMAP) |
| train | A100 / L40S / RTX Pro — one chunk per GPU |
| hierarchy merge (later) | 8×H200 |

The image is multi-arch-intended; GH200/Spark are arm64 — CUDA arch list is a
build arg.

## Portability: the cleavage points

Built on our stack, designed to run on anyone's. Three contracts keep it that
way — anything behind a contract is swappable without touching the rest:

1. **Camera contract** — two layers, so overfitting to one camera's quirks is
   structurally hard. A **profile** (`splatpipe/data/profiles/*.yaml`) holds
   every hardware fact: match rules, lens axes and coverage, telemetry source,
   default view density. A **driver** (`splatpipe/drivers/`) holds one
   projection and answers exactly two questions — *what pixel is this ray*,
   and *does this lens see it*. Everything downstream consumes the ingest
   layout: `images/camN/*.jpg` (pinhole views) + `frames.jsonl` + optional
   `geo.txt` (`camN/file.jpg lat lon alt`, WGS84).
   New camera, known projection → **a YAML file**. New projection → one class,
   three methods. Neither touches ingest, chunking, poses, or training.
2. **Stage contract** — every stage is a plain CLI over files on disk. The
   whole chain runs on a laptop. Fan-out needs no scheduler integration: point
   N workers at the same chunk directory and they coordinate through the queue,
   so bare metal, a Slurm array, plain k8s Jobs, or our LWS/devpod setup all
   work unmodified.
3. **Container contract** — `Dockerfile` takes `BASE_IMAGE` as a build arg:
   swap our `ai-dev-pod` for any CUDA-enabled torch image. The `deploy/`
   manifests are optional conveniences for our platform, not dependencies —
   nothing in `splatpipe/` imports or assumes them.
4. **Resource floor** — a single GPU on a 32GB-RAM box is a supported target,
   not a degraded one (empirically proven: that's our dev pod). Build
   concurrency auto-sizes from live CPU/RAM down to 1 job (single CUDA
   compile jobs peak ~9-12GB); the JIT/object caches live on persistent
   storage and resume across OOM kills. Big iron makes it faster, never
   required.

## Data sources

Licenses and required attributions for all of these live in [THIRD_PARTY.md](THIRD_PARTY.md).

- `just fetch-360` — real raw GoPro Max `.360` (CC BY 4.0, AMBIENT project / RITMO, Univ. of Oslo; doi:10.5281/zenodo.21611765); drives the smoke test
- [Mapillary](https://help.mapillary.com/hc/en-us/articles/360012674619-GoPro-MAX-Series) — car-mounted GoPro Max 360 sequences + GPS via Graph API (CC BY-SA 4.0: dev/testing only, not shipped assets)
- [gpmf-parser samples](https://github.com/gopro/gpmf-parser) — GoPro files with GPMF telemetry (Apache-2.0)
- [H3DGS toy dataset](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/example_dataset.zip) — optional benchmark ONLY (INRIA research/eval-only license, non-commercial)
- [Trek View .360 recipes](https://www.trekview.org/blog/using-ffmpeg-process-gopro-max-360/) + [max2sphere](https://github.com/trek-view/max2sphere) (Apache-2.0) — the references behind `splatpipe/eac.py`

## Roadmap

1. ✅ ingest / chunk / poses / train, torchrun-sharded
2. ✅ `.360` EAC ingest (validated on real Max footage; re-verify on Max 2 8K)
3. ✅ locality chunking + capture corridors + claim-based fan-out + merge
4. Transient masking (moving cars, capture-vehicle shadow) + sky masks
5. Mesh export: implemented (`splatpipe mesh`) but **unvalidated** — needs
   driving capture with real baseline; classic MVS is the fallback
6. LOD hierarchy over the merged tiles + per-image appearance embeddings
7. Export path: compressed splats (spz/sog); `splatpipe route`/`export` already
   emit stage centreline + road ribbon for point-to-point sim tracks
8. Lidar cross-check (Maryland lidar shares the ENU frame) for drift + collision
