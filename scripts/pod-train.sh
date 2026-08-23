#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Precompile gsplat's CUDA ops, then train a chunk. Usage:
#   scripts/pod-train.sh <chunk-dir> [max-steps]
#
# Why this exists: gsplat JIT-compiles its kernels on first import, and torch's
# cpp_extension defaults to EVERY arch the local torch build supports (on the
# NVIDIA containers that is sm_75..sm_120 — seven device compiles per file).
# The fused rasterizer kernels are ~10-20GB and ~1h of cicc EACH at that
# breadth, which OOMs or stalls any modest box. We compile for the installed
# GPU's arch only (a 7x cut) with cicc's low-memory mode, so a single-GPU
# 32GB machine can get through it.
set -euo pipefail

CHUNK=${1:?usage: pod-train.sh <chunk-dir> [max-steps]}
STEPS=${2:-15000}
GSPLAT_EXAMPLES=${GSPLAT_EXAMPLES:-/opt/gsplat/examples}
[ -d "$GSPLAT_EXAMPLES" ] || GSPLAT_EXAMPLES=/workspace/opt/gsplat/examples

# shellcheck disable=SC1091
source "$(dirname "$0")/gsplat-env.sh"   # shared build env — see that file for why

# Stage 1: build the extension alone (no trainer resident = max headroom).
python -c 'import gsplat; from gsplat import rasterization; print("[train] gsplat ops ready", gsplat.__version__)'

# Stage 2: train. Kernels are cached now, so this starts immediately.
exec python "$GSPLAT_EXAMPLES/simple_trainer.py" default \
    --data-dir "$CHUNK" --data-factor 1 \
    --result-dir "$CHUNK/splat" --max-steps "$STEPS" --save-ply
