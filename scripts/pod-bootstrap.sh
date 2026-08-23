# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
# Recover a dev pod after a container restart. Everything durable lives on the
# PVC (/workspace: venv, wheels, COLMAP at /workspace/opt/sfm, data, repos);
# this reinstalls only the ephemeral container-fs bits and wires the PATH.
set -euo pipefail

sudo chown "$(id -u):$(id -g)" /workspace 2>/dev/null || true
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libimage-exiftool-perl

if [ ! -d /workspace/venv ]; then
    python -m venv --system-site-packages /workspace/venv
fi
# shellcheck disable=SC1091
source /workspace/venv/bin/activate
export PIP_CACHE_DIR=/workspace/.pip-cache

# fast path: prebuilt wheels cached on the PVC (see README dev-pod section)
if ls /workspace/wheels/*.whl >/dev/null 2>&1; then
    pip install -q /workspace/wheels/*.whl
fi
pip install -q -e /workspace/gaussworks

echo 'export PATH=/workspace/opt/sfm/bin:$PATH' >> ~/.bashrc
echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
echo "[pod-bootstrap] done — colmap: $(command -v colmap || echo MISSING), venv active"
