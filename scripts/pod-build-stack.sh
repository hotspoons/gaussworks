#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the full SfM + training stack from source on a dev pod, onto the PVC.
# Written for a BARE base image: nothing but python3, uv, gcc and sudo is
# assumed. Idempotent: every stage checks for its own output. First run is
# ~60-90 min, dominated by the CUDA download, COLMAP and the gsplat kernels.
#
# Layout (all under /workspace, which is the only thing that survives):
#   opt/cuda      CUDA toolkit (runfile, --toolkitpath)   opt/exiftool  GPS9-capable exiftool
#   opt/sfm       COLMAP + GLOMAP (+ lib/bundled: their .so deps, so no apt at runtime)
#   opt/gsplat    gsplat source + examples (reference trainer)
#   venv          python env incl. torch cu130           apt-cache     .debs for offline bootstrap
#   downloads     runfiles/tarballs                       src           colmap / glomap trees
#
# Concurrency: this pod has no cgroup memory limit, so a compile burst hurts
# the NODE, not just the pod. CXX_JOBS / MAX_JOBS are sized from free RAM and
# can be overridden. History: 16 blind jobs OOM-killed a 32 GB pod three times.
set -euo pipefail

COLMAP_VER=${COLMAP_VER:-3.11.1}
# GLOMAP 1.0.0, NOT the latest: 1.2.0 vendors an Oct-2025 COLMAP whose rig
# model aborts writing a reconstruction built from a 3.11 database (LANDSCAPE).
GLOMAP_VER=${GLOMAP_VER:-1.0.0}
GSPLAT_REF=${GSPLAT_REF:-main}
EXIFTOOL_VER=${EXIFTOOL_VER:-13.44}     # 12.90+ required for GoPro GPS9
# CUDA runfile name is ARCH-SPECIFIC: NVIDIA suffixes the Grace/ARM build
# "_linux_sbsa". Getting this wrong 404s rather than failing usefully, and the
# GH200 fleet (gh200-1) is aarch64 while the original A100 pod was x86_64.
case "$(uname -m)" in
    aarch64|arm64) _cuda_plat=linux_sbsa ;;
    *)             _cuda_plat=linux ;;
esac
CUDA_RUNFILE=${CUDA_RUNFILE:-cuda_13.0.2_580.95.05_${_cuda_plat}.run}   # major must match the torch wheel
TORCH_SPEC=${TORCH_SPEC:-"torch==2.9.1 torchvision"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
CUDA_ARCHS=${CUDA_ARCHS:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')}
CUDA_ARCHS=${CUDA_ARCHS:-80}
W=/workspace; S=$W/src; P=$W/opt/sfm
mkdir -p $W/opt $W/src $W/downloads $W/logs $W/bin $W/tmp $W/apt-cache/archives/partial

if pgrep -c -x cicc >/dev/null 2>&1 || pgrep -c -x ptxas >/dev/null 2>&1; then
    echo "[build-stack] REFUSING to start: a CUDA build is already running." >&2; exit 1
fi
cores=$(nproc)
avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
cap() { local v=$1 lo=$2 hi=$3; [ "$v" -lt "$lo" ] && v=$lo; [ "$v" -gt "$hi" ] && v=$hi; echo "$v"; }
CXX_JOBS=${CXX_JOBS:-$(cap $(( avail_gb / 4 )) 1 "$cores")}      # C++ TUs ~2 GB, leave headroom
export MAX_JOBS=${MAX_JOBS:-$(cap $(( avail_gb / 16 )) 1 4)}      # cicc ~9 GB each at -O1
echo "[build-stack] cores=$cores avail=${avail_gb}GB arch=$CUDA_ARCHS -> CXX_JOBS=$CXX_JOBS MAX_JOBS=$MAX_JOBS"

# --- env file first, so every later step (and every shell) agrees -------------
cp "$(dirname "$0")/env.sh" $W/env.sh
cp "$(dirname "$0")/claude-sync.sh" $W/bin/claude-sync.sh && chmod +x $W/bin/claude-sync.sh
grep -q 'source /workspace/env.sh' ~/.bashrc 2>/dev/null \
    || echo '[ -f /workspace/env.sh ] && source /workspace/env.sh' >> ~/.bashrc
[ -f $W/.gitconfig ] || cat > $W/.gitconfig <<'RC'
[user]
	name = Rich Siomporas
	email = richard.siomporas@patapsco.ai
[credential]
	helper = store --file=/workspace/.git-credentials
[safe]
	directory = *
[pull]
	ff = only
RC

# --- apt: install AND keep the .debs on the PVC for offline bootstraps -------
PKGS="ffmpeg rsync tmux cmake ninja-build build-essential git
      libboost-program-options-dev libboost-graph-dev libboost-system-dev
      libboost-filesystem-dev libboost-test-dev libeigen3-dev libfreeimage-dev
      libmetis-dev libgoogle-glog-dev libgflags-dev libsqlite3-dev libceres-dev
      libflann-dev libsuitesparse-dev libcgal-dev libglew-dev"
sudo apt-get update -qq
# shellcheck disable=SC2086
sudo apt-get install -y -qq -o Dir::Cache::archives=$W/apt-cache/archives \
    -o APT::Keep-Downloaded-Packages=true $PKGS
sudo chown -R "$(id -u):$(id -g)" $W/apt-cache

# --- CUDA toolkit onto the PVC ------------------------------------------------
# If the base image already ships a toolkit of the right major version, adopt
# it instead of spending 4 GB and ~15 min re-installing one. Adopt by TESTING
# nvcc, not by testing that a path exists -- a dangling symlink here would fail
# later, inside a COLMAP build, as something that looks like a CMake problem.
if [ ! -x $W/opt/cuda/bin/nvcc ] && [ -x "${SYSTEM_CUDA:-/usr/local/cuda}/bin/nvcc" ]; then
    _sys=${SYSTEM_CUDA:-/usr/local/cuda}
    _want=$(echo "$CUDA_RUNFILE" | sed -E 's/cuda_([0-9]+)\..*/\1/')
    _have=$("$_sys/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p')
    if [ "$_have" = "$_want" ]; then
        echo "[build-stack] adopting image CUDA $_have at $_sys (no runfile needed)"
        mkdir -p $W/opt && ln -sfn "$(readlink -f "$_sys")" $W/opt/cuda
        $W/opt/cuda/bin/nvcc --version >/dev/null || { echo "[build-stack] adopted nvcc does not run" >&2; exit 1; }
    else
        echo "[build-stack] image CUDA $_have != required $_want; installing runfile"
    fi
fi
if [ ! -x $W/opt/cuda/bin/nvcc ]; then
    ver=$(echo "$CUDA_RUNFILE" | sed -E 's/cuda_([0-9.]+)_.*/\1/')
    f=$W/downloads/$CUDA_RUNFILE
    [ -s "$f" ] || curl -fL --retry 5 -C - -o "$f" \
        "https://developer.download.nvidia.com/compute/cuda/${ver}/local_installers/${CUDA_RUNFILE}"
    # --tmpdir on the PVC: the self-extractor needs ~5 GB of scratch
    sh "$f" --silent --toolkit --toolkitpath=$W/opt/cuda --no-drm --no-man-page \
        --override --tmpdir=$W/tmp
fi
# shellcheck disable=SC1091
source $W/env.sh
nvcc --version | tail -1

# --- python: venv with torch, on the PVC ---------------------------------------
export UV_CACHE_DIR=$W/.uv-cache
[ -x $W/venv/bin/python ] || uv venv $W/venv --python 3.12 --seed
$W/venv/bin/python -c "import torch" 2>/dev/null \
    || uv pip install --python $W/venv/bin/python $TORCH_SPEC --index-url "$TORCH_INDEX"
# shellcheck disable=SC1091
source $W/venv/bin/activate

# --- ExifTool (GitHub mirror: exiftool.org only serves the newest release) ------
if [ ! -x $W/opt/exiftool/exiftool ]; then
    curl -fsSL -o $W/downloads/exiftool-$EXIFTOOL_VER.tar.gz \
        "https://github.com/exiftool/exiftool/archive/refs/tags/$EXIFTOOL_VER.tar.gz"
    rm -rf $W/opt/exiftool && mkdir -p $W/opt/exiftool
    tar xzf $W/downloads/exiftool-$EXIFTOOL_VER.tar.gz -C $W/opt/exiftool --strip-components=1
fi
[ "$(exiftool -listx 2>/dev/null | grep -c GPS9)" -gt 0 ] \
    || { echo "[build-stack] FATAL: exiftool has no GPS9 support" >&2; exit 1; }

# --- COLMAP (CUDA SIFT), headless ----------------------------------------------
if [ ! -x $P/bin/colmap ]; then
    [ -d $S/colmap ] || git clone --depth 1 -b "$COLMAP_VER" https://github.com/colmap/colmap $S/colmap
    cmake -S $S/colmap -B $S/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_PREFIX_PATH=$P -DCMAKE_INSTALL_PREFIX=$P
    cmake --build $S/colmap/build --target install -j "$CXX_JOBS"
fi

# --- GLOMAP: global SfM -------------------------------------------------------------
if [ ! -x $P/bin/glomap ]; then
    [ -d $S/glomap ] || git clone --depth 1 -b "$GLOMAP_VER" https://github.com/colmap/glomap $S/glomap
    cmake -S $S/glomap -B $S/glomap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
        -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_PREFIX_PATH=$P -DCMAKE_INSTALL_PREFIX=$P
    cmake --build $S/glomap/build --target install -j "$CXX_JOBS"
fi

# --- bundle the binaries' shared libs, so a restart needs no apt to run them ----
mkdir -p $P/lib/bundled
for b in $P/bin/colmap $P/bin/glomap; do
    ldd "$b" | awk '/=> \//{print $3}' \
      | grep -vE "^$W/opt/cuda|/(libc|libm|libdl|libpthread|librt|libstdc\+\+|libgcc_s|ld-linux)[.-]" \
      | while read -r l; do cp -Lun "$l" $P/lib/bundled/ 2>/dev/null || true; done
done
echo "[build-stack] bundled $(ls $P/lib/bundled | wc -l) shared libs into $P/lib/bundled"

# --- gsplat + reference trainer deps -------------------------------------------------
[ -d $W/opt/gsplat ] || git clone --recursive https://github.com/nerfstudio-project/gsplat $W/opt/gsplat
git -C $W/opt/gsplat checkout -q "$GSPLAT_REF"
grep -vE "git\+" $W/opt/gsplat/examples/requirements.txt > $W/tmp/gsplat-req.txt
uv pip install --python $W/venv/bin/python -r $W/tmp/gsplat-req.txt jaxtyping nvtx "rich>=12" ninja cupy-cuda13x
python -c "import gsplat" 2>/dev/null \
    || uv pip install --python $W/venv/bin/python --no-build-isolation $W/opt/gsplat
# git deps need torch at build time -> one by one, non-fatal (optional trainer features)
while read -r dep; do
    uv pip install --python $W/venv/bin/python --no-build-isolation "$dep" \
        || echo "[build-stack] optional dep failed: $dep"
done < <(grep -E "git\+" $W/opt/gsplat/examples/requirements.txt | sed 's/^# *//')
mkdir -p $W/wheels
ls $W/wheels/gsplat-*.whl >/dev/null 2>&1 \
    || uv pip wheel --no-build-isolation --no-deps -w $W/wheels $W/opt/gsplat 2>/dev/null || true

uv pip install --python $W/venv/bin/python -e /workspace/gaussworks

# --- verify, loudly ---------------------------------------------------------------------
python -c "import gsplat, imageio, viser; print('[build-stack] OK gsplat', gsplat.__version__)"
echo "[build-stack] colmap:   $(colmap -h 2>&1 | sed -n 2p)"
echo "[build-stack] glomap:   $(glomap -h 2>&1 | sed -n 3p)"
echo "[build-stack] exiftool: $(exiftool -ver)"
splatpipe profiles
bash "$(dirname "$0")/pod-bootstrap.sh"
echo "[build-stack] done. New shells pick up the env from /workspace/env.sh."
