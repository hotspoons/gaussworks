#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Recover a dev pod after a container restart. Everything durable lives on the
# PVC (/workspace: venv, wheels, COLMAP at /workspace/opt/sfm, data, repos);
# this reinstalls only the ephemeral container-fs bits and wires the PATH.
set -euo pipefail

sudo chown "$(id -u):$(id -g)" /workspace 2>/dev/null || true
sudo apt-get update -qq
# Everything here lives on the CONTAINER filesystem, so it vanishes on every
# restart while /workspace survives -- including the shared libraries the
# PVC-installed COLMAP was linked against. Reinstalling them is most of what
# this script is for.
sudo apt-get install -y -qq ffmpeg libimage-exiftool-perl \
    libx11-6 libgl1 libgomp1 \
    libopengl0 libglew2.2 libfreeimage3 libmetis5 libceres4 \
    libboost-program-options1.83.0 libboost-graph1.83.0 \
    libgoogle-glog0v6 libgflags2.2 libflann1.9 libsuitesparse-dev \
    || sudo apt-get install -y -qq ffmpeg libimage-exiftool-perl \
        libx11-6 libgl1 libgomp1 libopengl0 libglew2.2 libfreeimage3 \
        libmetis5 libgoogle-glog0v6 libgflags2.2   # older/newer name drift

if [ ! -d /workspace/venv ]; then
    python -m venv --system-site-packages /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
export PIP_CACHE_DIR=/workspace/.pip-cache
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions   # gsplat JITs CUDA ops at first import; keep the cache on the PVC

# fast path: prebuilt wheels cached on the PVC (see README dev-pod section)
if ls /workspace/wheels/*.whl >/dev/null 2>&1; then
    pip install -q /workspace/wheels/*.whl
fi
pip install -q -e /workspace/gaussworks


# ExifTool: the distro package is too old. GoPro switched the GPMF GPS payload
# from GPS5 to GPS9 with the HERO11 generation (the MAX 2 writes GPS9), and an
# exiftool without GPS9 support parses the file, reports every other stream,
# and silently returns no GPS -- which downstream is indistinguishable from a
# camera that never got a fix. Ubuntu 24.04 ships 12.76; we need 12.90+.
EXIFTOOL_VERSION=13.44
if [ ! -x /workspace/opt/exiftool/exiftool ]; then
    mkdir -p /workspace/opt && cd /workspace/opt
    curl -fsSL -o et.tgz "https://exiftool.org/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz"
    rm -rf exiftool Image-ExifTool-* && tar xzf et.tgz && rm et.tgz
    mv "Image-ExifTool-${EXIFTOOL_VERSION}" exiftool
fi

echo 'export PATH=/workspace/opt/exiftool:/workspace/opt/sfm/bin:$PATH' >> ~/.bashrc
echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
echo 'export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions' >> ~/.bashrc
export PATH=/workspace/opt/exiftool:/workspace/opt/sfm/bin:$PATH
echo "[pod-bootstrap] done — colmap: $(command -v colmap || echo MISSING), \
exiftool: $(exiftool -ver 2>/dev/null || echo MISSING), venv active"
