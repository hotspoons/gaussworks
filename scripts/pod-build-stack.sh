#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the full SfM+training stack from source on a dev pod, onto the PVC.
# Mirrors the Dockerfile stages 1:1 — this is the interactive twin of the
# image build, for pods running the plain ai-dev-pod base. Idempotent-ish:
# skips COLMAP if already installed. ~40-60 min first run.
#
# Concurrency and CUDA arch come from scripts/gsplat-env.sh; override with
# CXX_JOBS / MAX_JOBS env vars. History: 16 blind jobs OOM-killed the pod.
set -euo pipefail

COLMAP_VER=${COLMAP_VER:-3.11.1}
GSPLAT_REF=${GSPLAT_REF:-main}
CUDA_ARCHS=${CUDA_ARCHS:-80}            # A100; add 89;90;120 for other pools
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export PIP_CACHE_DIR=/workspace/.pip-cache
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions   # gsplat JITs CUDA ops at first import; keep the cache on the PVC

# --- concurrency from live resources ------------------------------------------
# Measured on gsplat: a single ptxas peaks ~9GB RES; plain C++ TUs ~2GB. Size
# each pool by RAM and cap at core count, instead of guessing. MemAvailable is
# node-level when the container has no memory limit — which is exactly the
# budget that matters, since the kernel OOM-killer acts node-wide.
# refuse to stack builds: overlapping compile bursts are how nodes die
if pgrep -c -f "cicc|ptxas|cudafe|nvcc" >/dev/null 2>&1; then
    echo "[build-stack] REFUSING to start: a CUDA build is already running (pgrep cicc/ptxas/nvcc)." >&2
    exit 1
fi

cores=$(nproc)
avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
cap() { local v=$1 lo=$2 hi=$3; [ "$v" -lt "$lo" ] && v=$lo; [ "$v" -gt "$hi" ] && v=$hi; echo "$v"; }
CXX_JOBS=${CXX_JOBS:-$(cap $(( avail_gb / 2 )) 1 "$cores")}
export MAX_JOBS=${MAX_JOBS:-$(cap $(( avail_gb / 9 )) 1 "$cores")}   # CUDA ext builds
echo "[build-stack] cores=$cores avail=${avail_gb}GB -> CXX_JOBS=$CXX_JOBS MAX_JOBS=$MAX_JOBS"

# --- build deps (container-fs, cheap to redo after restarts) -----------------
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build build-essential git \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgflags-dev libsqlite3-dev libceres-dev libflann-dev \
    libsuitesparse-dev libcgal-dev libglew-dev

# --- venv on the PVC ----------------------------------------------------------
[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# --- CUDA COLMAP -> /workspace/opt/sfm ---------------------------------------
if [ ! -x /workspace/opt/sfm/bin/colmap ]; then
    [ -d /workspace/src/colmap ] || \
        git clone --depth 1 -b "${COLMAP_VER}" https://github.com/colmap/colmap /workspace/src/colmap
    cmake -S /workspace/src/colmap -B /workspace/src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/workspace/opt/sfm
    cmake --build /workspace/src/colmap/build --target install -j "$CXX_JOBS"
fi

# --- gsplat + reference trainer deps -----------------------------------------
[ -d /workspace/opt/gsplat ] || \
    git clone --recursive https://github.com/nerfstudio-project/gsplat /workspace/opt/gsplat
cd /workspace/opt/gsplat && git checkout -q "${GSPLAT_REF}"
pip install --no-build-isolation .
# plain PyPI deps first; the git deps need torch at build time, so they go
# one-by-one with --no-build-isolation (a failure there is non-fatal: they
# back optional trainer features like bilateral grids)
grep -vE "git\+" examples/requirements.txt | pip install -r /dev/stdin
while read -r dep; do
    pip install --no-build-isolation "$dep" || echo "[build-stack] optional dep failed: $dep"
done < <(grep -E "git\+" examples/requirements.txt)

# --- wheel cache: wiped-venv recovery in seconds, reuses the build tree ------
mkdir -p /workspace/wheels
pip wheel --no-build-isolation --no-deps -w /workspace/wheels /workspace/opt/gsplat

pip install -e /workspace/gaussworks
python -c "import gsplat, imageio, viser; print('[build-stack] OK gsplat', gsplat.__version__)"
echo "[build-stack] colmap: /workspace/opt/sfm/bin/colmap ($(/workspace/opt/sfm/bin/colmap help 2>&1 | head -1))"
