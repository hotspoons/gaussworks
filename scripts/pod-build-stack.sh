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
# GLOMAP 1.0.0, NOT the latest. 1.2.0 vendors a COLMAP from Oct 2025 that
# models multi-camera setups as rigs; it opens a COLMAP 3.11 database, migrates
# in empty rigs/rig_sensors/frames tables, completes an entire reconstruction,
# and then ABORTS writing the model (`Check failed: existing_rig.RefSensorId()
# == rig.RefSensorId()`). Match the COLMAP generation instead. NB: GLOMAP tags
# carry no "v" prefix, and its vendored COLMAP needs GUI_ENABLED=OFF or it
# demands Qt5.
GLOMAP_VER=${GLOMAP_VER:-1.0.0}
GSPLAT_REF=${GSPLAT_REF:-main}
EXIFTOOL_VER=${EXIFTOOL_VER:-13.44}     # 12.90+ required for GoPro GPS9
CERES_CUDA=${CERES_CUDA:-0}             # 1 = build Ceres with CUDA (see below)
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
    libboost-filesystem-dev libboost-test-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgflags-dev libsqlite3-dev libceres-dev libflann-dev \
    libsuitesparse-dev libcgal-dev libglew-dev

# --- venv on the PVC ----------------------------------------------------------
[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# --- ExifTool from upstream ---------------------------------------------------
# The distro package (12.76 on Ubuntu 24.04) predates GoPro's GPS9 payload,
# which the MAX 2 writes. It parses the file, reports every other telemetry
# stream, and returns NO GPS -- silently disabling geo alignment and locality
# chunking. Non-negotiable: install a current one.
if [ ! -x /workspace/opt/exiftool/exiftool ]; then
    mkdir -p /workspace/opt && cd /workspace/opt
    curl -fsSL -o et.tgz "https://exiftool.org/Image-ExifTool-${EXIFTOOL_VER}.tar.gz"
    rm -rf exiftool "Image-ExifTool-${EXIFTOOL_VER}" && tar xzf et.tgz && rm et.tgz
    mv "Image-ExifTool-${EXIFTOOL_VER}" exiftool
fi
/workspace/opt/exiftool/exiftool -ver
[ "$(/workspace/opt/exiftool/exiftool -listx 2>/dev/null | grep -c GPS9)" -gt 0 ] \
    || { echo "[build-stack] FATAL: exiftool has no GPS9 support" >&2; exit 1; }

# --- optional: Ceres with CUDA -----------------------------------------------
# Only matters when the INCREMENTAL mapper runs (i.e. GLOMAP is unavailable):
# it is what makes COLMAP's --Mapper.ba_use_gpu do anything, since the distro
# libceres has no CUDA. poses.py probes colmap's linkage before passing the
# flag, so leaving this off is safe -- it just stays on the CPU. Adds ~15 min.
if [ "$CERES_CUDA" = "1" ] && [ ! -f /workspace/opt/sfm/lib/cmake/Ceres/CeresConfig.cmake ]; then
    [ -d /workspace/src/ceres ] || git clone --depth 1 -b 2.2.0 \
        https://github.com/ceres-solver/ceres-solver /workspace/src/ceres
    cmake -S /workspace/src/ceres -B /workspace/src/ceres/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX=/workspace/opt/sfm
    cmake --build /workspace/src/ceres/build --target install -j "$CXX_JOBS"
fi

# --- CUDA COLMAP -> /workspace/opt/sfm ---------------------------------------
if [ ! -x /workspace/opt/sfm/bin/colmap ]; then
    [ -d /workspace/src/colmap ] || \
        git clone --depth 1 -b "${COLMAP_VER}" https://github.com/colmap/colmap /workspace/src/colmap
    cmake -S /workspace/src/colmap -B /workspace/src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF -DCMAKE_PREFIX_PATH=/workspace/opt/sfm \
        -DCMAKE_INSTALL_PREFIX=/workspace/opt/sfm
    cmake --build /workspace/src/colmap/build --target install -j "$CXX_JOBS"
fi

# --- GLOMAP: global SfM ------------------------------------------------------
# The mapper is the longest stage in the pipeline and COLMAP's incremental one
# is CPU-only: measured 4h53m on a 3,870-image chunk, against 73 min for all
# the GPU feature/matching stages combined. GLOMAP solves all images at once.
# poses.py picks it up automatically whenever it is on PATH -- so a pod without
# it silently runs the slow path. VERIFY BY RUNNING A REAL MAPPING: `glomap -h`
# happily prints "compiled with CUDA!" on a binary that cannot finish one.
if [ ! -x /workspace/opt/sfm/bin/glomap ]; then
    [ -d /workspace/src/glomap ] || \
        git clone --depth 1 -b "${GLOMAP_VER}" https://github.com/colmap/glomap /workspace/src/glomap
    cmake -S /workspace/src/glomap -B /workspace/src/glomap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF \
        -DCMAKE_PREFIX_PATH=/workspace/opt/sfm \
        -DCMAKE_INSTALL_PREFIX=/workspace/opt/sfm
    cmake --build /workspace/src/glomap/build --target install -j "$CXX_JOBS"
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

# --- verify, loudly ----------------------------------------------------------
export PATH=/workspace/opt/sfm/bin:/workspace/opt/exiftool:$PATH
python -c "import gsplat, imageio, viser; print('[build-stack] OK gsplat', gsplat.__version__)"
echo "[build-stack] colmap:   $(colmap -h 2>&1 | sed -n 2p)"
echo "[build-stack] glomap:   $(glomap -h 2>&1 | sed -n 3p)"
echo "[build-stack] exiftool: $(exiftool -ver)"
splatpipe profiles
cat >> ~/.bashrc <<'RC'
export PATH=/workspace/opt/sfm/bin:/workspace/opt/exiftool:$PATH
export EXIFTOOL=/workspace/opt/exiftool/exiftool
export GSPLAT_EXAMPLES=/workspace/opt/gsplat/examples
source /workspace/gaussworks/scripts/gsplat-env.sh
source /workspace/venv/bin/activate
RC
echo "[build-stack] done. New shells pick up PATH/env from ~/.bashrc."
