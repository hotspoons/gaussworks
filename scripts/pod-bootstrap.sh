#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Recover a dev pod after a container restart -- seconds, offline-capable.
#
# Everything durable lives on the PVC (/workspace): CUDA toolkit, venv (with
# torch), COLMAP/GLOMAP + their bundled shared libs, exiftool, gsplat, repos,
# caches, and a .deb cache. The container filesystem (home dir, apt state,
# ~/.bashrc, ~/.claude) is thrown away on every restart. This script puts back
# the few ephemeral bits and wires every shell to /workspace/env.sh.
#
# Fresh PVC? Run scripts/pod-build-stack.sh instead (this is called at its end).
set -euo pipefail

sudo chown "$(id -u):$(id -g)" /workspace 2>/dev/null || true
mkdir -p /workspace/logs /workspace/bin

# --- 1. persistent environment, sourced by every shell -----------------------
if [ ! -f /workspace/env.sh ]; then
    cp "$(dirname "$0")/env.sh" /workspace/env.sh
fi
grep -q 'source /workspace/env.sh' ~/.bashrc 2>/dev/null \
    || echo '[ -f /workspace/env.sh ] && source /workspace/env.sh' >> ~/.bashrc
# shellcheck disable=SC1091
source /workspace/env.sh

# --- 2. apt packages, from the PVC .deb cache when possible -------------------
# ffmpeg (with NVDEC) and the build tools live on the container fs. The COLMAP
# and GLOMAP binaries carry their own shared libs in /workspace/opt/sfm/lib/
# bundled (see pod-build-stack.sh), so they do not need apt at all any more.
PKGS="ffmpeg rsync tmux cmake ninja-build build-essential git
      libboost-program-options-dev libboost-graph-dev libboost-system-dev
      libboost-filesystem-dev libboost-test-dev libeigen3-dev libfreeimage-dev
      libmetis-dev libgoogle-glog-dev libgflags-dev libsqlite3-dev libceres-dev
      libflann-dev libsuitesparse-dev libcgal-dev libglew-dev"
CACHE=/workspace/apt-cache/archives
if ! command -v ffmpeg >/dev/null || ! command -v cmake >/dev/null; then
    if ls "$CACHE"/*.deb >/dev/null 2>&1; then
        # offline fast path: every dependency was cached on the PVC at build time
        sudo dpkg -i --force-depends "$CACHE"/*.deb >/workspace/logs/dpkg.log 2>&1 \
            || echo "[pod-bootstrap] dpkg had complaints (see logs/dpkg.log); fixing up via apt"
    fi
    sudo apt-get update -qq || true
    # shellcheck disable=SC2086
    sudo apt-get install -y -qq -o Dir::Cache::archives="$CACHE" $PKGS
    sudo chown -R "$(id -u):$(id -g)" /workspace/apt-cache 2>/dev/null || true
fi

# --- 3. python: venv is on the PVC; just re-link the editable install ----------
if [ -x /workspace/venv/bin/python ]; then
    uv pip install -q --python /workspace/venv/bin/python -e /workspace/gaussworks 2>/dev/null \
        || /workspace/venv/bin/pip install -q -e /workspace/gaussworks
else
    echo "[pod-bootstrap] no /workspace/venv -- run scripts/pod-build-stack.sh" >&2
fi

# --- 4. Claude Code state: restore from the PVC, then keep it synced ----------
if [ -x /workspace/bin/claude-sync.sh ]; then
    /workspace/bin/claude-sync.sh start
fi

# --- 5. report --------------------------------------------------------------------
echo "[pod-bootstrap] done"
echo "  nvcc:     $(command -v nvcc || echo MISSING)"
echo "  colmap:   $(colmap -h 2>&1 | sed -n 2p || echo MISSING)"
echo "  glomap:   $(glomap -h 2>&1 | head -1 || echo MISSING)"
echo "  exiftool: $(exiftool -ver 2>/dev/null || echo MISSING)  (GPS9 tags: $(exiftool -listx 2>/dev/null | grep -c GPS9))"
echo "  ffmpeg:   $(ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -c cuda) cuda hwaccel"
echo "  python:   $(command -v python)  torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
