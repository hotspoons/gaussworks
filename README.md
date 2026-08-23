# gaussworks — car-mounted 360 capture → gaussian splat pipeline

Turns wide-angle / 360 driving (and hiking) footage into chunked, geo-aligned
3D Gaussian Splat reconstructions. First target: the back-road network between
Crofton and Annapolis (see `configs/md-backroads.yaml`).

Canonical repo: https://github.com/hotspoons/gaussworks (mirrored on internal GitLab)
(worked on locally as a clone under `trailworks/ext/`). Self-contained: own
justfile, package, and container; Apache-2.0 (see LICENSE and THIRD_PARTY.md).
Outputs are geo-aligned (ENU), so they can eventually feed trailworks as a
detail layer.

## Pipeline

```
video (.mp4 equirect / .360 EAC*)        Mapillary sequences (equirect + GPS)
        │                                         │
        ▼                                         ▼
  [1] ingest ──────────────────────────► frames.jsonl + images/camN/*.jpg + geo.txt
        GPMF GPS via exiftool, distance-spaced frame pick (sharpest per window),
        equirect → K virtual pinhole views, EXIF GPS written into each jpg
        │
        ▼
  [2] chunk        overlapping ~200m segments along the GPS track
        │
        ▼
  [3] poses        per chunk: COLMAP features → spatial matching (GPS priors)
        │            → mapper (GLOMAP if present) → model_aligner to ENU
        ▼
  [4] train        per chunk: gsplat trainer, one GPU per chunk,
        │            sharded across ranks under torchrun / devpod launch
        ▼
     chunks/chunk_NNN/splat/*.ply   (merge/LOD hierarchy: next milestone)
```

\* `.360` (GoPro EAC, two-track) ingest is native: `splatpipe/eac.py` remaps
EAC -> pinhole in one resample (no equirect intermediate, no patched ffmpeg),
validated on a real GoPro Max file (`just fetch-360`). Max 2 8K files may use
a new track size — eyeball `eac_to_equirect()` output before trusting it.

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
.venv/bin/splatpipe ingest data/yt-run/video.mp4 --out data/run1 --projection equirect
.venv/bin/splatpipe chunk  --frames data/run1
.venv/bin/splatpipe poses  --chunks data/run1/chunks
.venv/bin/splatpipe train  --chunks data/run1/chunks
```

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
sudo apt-get install -y ffmpeg libimage-exiftool-perl   # ingest system deps
# iterate: edit anywhere, push, then here:
git pull --ff-only
```

## On the cluster (Zipspace)

Build/push the image, deploy `deploy/zipspace.yaml`, connect to the leader:

```bash
just image-build && just image-push
kubectl apply -f deploy/zipspace.yaml
```

Everything under `/workspace` is the shared RWX PVC. Stages 1–3 are
embarrassingly parallel shell work; stage 4 shards chunks across ranks, so on
the leader:

```bash
devpod launch python -m splatpipe.cli train --chunks /workspace/data/run1/chunks
```

Each rank claims `chunks[RANK::WORLD_SIZE]` and pins itself to `LOCAL_RANK`'s
GPU — a 4-node × 4-GPU group trains 16 chunks at a time. `poses` can be run
the same way (`devpod launch python -m splatpipe.cli poses ...` shards too).

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

1. **Camera contract** — every source reduces to the ingest layout:
   `images/camN/*.jpg` (pinhole views) + `frames.jsonl` (one record per
   capture position) + optional `geo.txt` (`camN/file.jpg lat lon alt`,
   WGS84). Supported today: GoPro `.360` (native EAC), **any stitched
   equirectangular video** — which covers Insta360, Qoocam, and most no-name
   360 cameras via their export apps (`--projection equirect`) — flat/pinhole
   video, image folders, and Mapillary sequences. Adding native Insta360
   `.insv` (dual fisheye) or any other format = one new source module writing
   this layout; nothing downstream knows or cares.
2. **Stage contract** — every stage is a plain CLI over files on disk. The
   whole chain runs on a laptop. Sharding is opt-in via standard `RANK` /
   `WORLD_SIZE` env vars (default 0/1), so bare metal, a Slurm array,
   plain k8s Jobs, or our LWS/devpod setup all work unmodified.
3. **Container contract** — `Dockerfile` takes `BASE_IMAGE` as a build arg:
   swap our `ai-dev-pod` for any CUDA-enabled torch image. The `deploy/`
   manifests are optional conveniences for our platform, not dependencies —
   nothing in `splatpipe/` imports or assumes them.

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
3. Transient masking (moving cars, capture-vehicle shadow) + sky masks
4. H3DGS hierarchy merge + per-image appearance embeddings (exposure drift)
5. Export path: compressed splats (spz/sog) + collision mesh from road centerline
