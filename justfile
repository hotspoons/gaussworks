# gaussworks — task recipes (https://just.systems)
# Self-contained: run from splats/, does not use the trailworks root justfile.

set shell := ["bash", "-euo", "pipefail", "-c"]

image := "harbor.tools.basedweights.com/patapsco.ai/gaussworks"
version := "0.1.0"

# list recipes
default:
    @just --list --unsorted

# --- setup -------------------------------------------------------------------

# create venv and install splatpipe editable
setup:
    if command -v uv >/dev/null; then uv venv --allow-existing && uv pip install -e .; \
    else python -m venv .venv && .venv/bin/pip install -q -e .; fi

# --- test data ---------------------------------------------------------------

# OPTIONAL benchmark. LICENSE WARNING: INRIA research/evaluation-only,
# non-commercial — NOT Apache/MIT-equivalent (see THIRD_PARTY.md). Nothing in
# the pipeline needs it (smoke uses the CC BY 4.0 .360 sample instead); never
# derive shipped assets from it.
# INRIA hierarchical-3d-gaussians toy dataset (1500 imgs, 2 chunks, ~5GB)
fetch-toy:
    mkdir -p data/toy
    curl -L --fail -C - -o data/toy/example_dataset.zip \
      https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/example_dataset.zip
    cd data/toy && unzip -n example_dataset.zip

# real raw GoPro Max .360 (3.8GB), pulled by byte range from inside Zenodo
# record 21611765's format-exemplars.zip (members are stored uncompressed, so
# a ranged GET is a bit-exact extraction).
# License: CC BY 4.0 — attribution: Guo, Riaz & Jensenius, "360-Degree Camera
# Comparison Dataset" (AMBIENT project, RITMO, University of Oslo),
# doi:10.5281/zenodo.21611765. See THIRD_PARTY.md.
fetch-360:
    mkdir -p data/samples
    curl -L -C - -r 162-4000379350 -o data/samples/GS010513.360 \
      "https://zenodo.org/api/records/21611765/files/format-exemplars.zip/content"
    echo "27ee5ac68feb0bb6ff57ca3d32602ac5  data/samples/GS010513.360" | md5sum -c -

# GoPro samples with real GPMF telemetry (incl. max-360mode.mp4) for
# GPS-extraction tests. License: Apache-2.0 (gopro/gpmf-parser).
fetch-sample-360:
    mkdir -p data/samples
    if [ ! -d data/samples/gpmf-parser ]; then \
      git clone --depth 1 https://github.com/gopro/gpmf-parser data/samples/gpmf-parser; fi
    ls -la data/samples/gpmf-parser/samples/

# download a YouTube 360 video (equirect, heavy compression — plumbing tests only)
fetch-yt URL OUT:
    mkdir -p {{OUT}}
    uvx yt-dlp -f "bv*+ba/b" -o "{{OUT}}/video.%(ext)s" --remux-video mp4 "{{URL}}"

# pull car-mounted 360 sequences from Mapillary (needs MAPILLARY_TOKEN; BBOX=w,s,e,n)
mapillary BBOX OUT:
    .venv/bin/splatpipe mapillary --bbox "{{BBOX}}" --out "{{OUT}}"

# --- pipeline (local) --------------------------------------------------------

# end-to-end sanity check on the real .360 sample (fetch-360 first);
# runs ingest -> chunk -> poses, prints the train command for a GPU box
smoke:
    .venv/bin/splatpipe smoke --out data/smoke

# --- container / cluster -----------------------------------------------------

# build the pipeline image (COLMAP+GLOMAP build takes a while the first time)
image-build:
    docker build -t {{image}}:{{version}} -t {{image}}:latest .

# push to harbor (needs user creds: docker login harbor.tools.basedweights.com)
image-push:
    docker push {{image}}:{{version}}
    docker push {{image}}:latest

# deploy the zipspace LWS group
deploy:
    kubectl apply -f deploy/zipspace.yaml
