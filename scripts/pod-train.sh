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

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/workspace/.torch_extensions}

# Arch of the GPU we actually have, e.g. "8.0" -> only sm_80 gets built.
# NOTE: the NVIDIA pytorch containers EXPORT TORCH_CUDA_ARCH_LIST with every
# supported arch ("7.5 8.0 8.6 9.0 10.0 12.0+PTX"), so we must override it
# rather than defer to it. Set GAUSSWORKS_ARCH to pin a list by hand (e.g.
# building a fat cache for a mixed fleet).
if [ -n "${GAUSSWORKS_ARCH:-}" ]; then
    TORCH_CUDA_ARCH_LIST=$GAUSSWORKS_ARCH
else
    TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "no GPU detected; set GAUSSWORKS_ARCH" >&2; exit 1; }
fi
export TORCH_CUDA_ARCH_LIST

# cicc peaks ~9-20GB per job on the fused kernels; -O1 cuts that hard for a
# negligible runtime difference. Size jobs off free RAM at ~10GB each.
avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
jobs=$(( avail_gb / 10 )); [ "$jobs" -lt 1 ] && jobs=1
export MAX_JOBS=${MAX_JOBS:-$jobs}
export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-"-Xcicc -O1"}

echo "[train] arch=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS avail=${avail_gb}GB cache=$TORCH_EXTENSIONS_DIR"

# Stage 1: build the extension alone (no trainer resident = max headroom).
python -c 'import gsplat; from gsplat import rasterization; print("[train] gsplat ops ready", gsplat.__version__)'

# Stage 2: train. Kernels are cached now, so this starts immediately.
exec python "$GSPLAT_EXAMPLES/simple_trainer.py" default \
    --data-dir "$CHUNK" --data-factor 1 \
    --result-dir "$CHUNK/splat" --max-steps "$STEPS" --save-ply
