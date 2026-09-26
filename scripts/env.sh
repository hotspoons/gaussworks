# gaussworks pod environment -- everything durable lives on the PVC (/workspace).
# Sourced from ~/.bashrc by scripts/pod-bootstrap.sh; safe to source repeatedly.
export WORKSPACE=/workspace
export CUDA_HOME=/workspace/opt/cuda
export PATH=/workspace/opt/cuda/bin:/workspace/opt/sfm/bin:/workspace/opt/exiftool:/workspace/opt/ffmpeg/bin:$PATH
export LD_LIBRARY_PATH=/workspace/opt/cuda/lib64:/workspace/opt/sfm/lib:/workspace/opt/sfm/lib/bundled${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export EXIFTOOL=/workspace/opt/exiftool/exiftool
export GSPLAT_EXAMPLES=/workspace/opt/gsplat/examples
export UV_CACHE_DIR=/workspace/.uv-cache
export PIP_CACHE_DIR=/workspace/.pip-cache
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions
export TORCH_HOME=/workspace/.torch-home          # torchvision/LPIPS weights, otherwise re-downloaded after every restart
# git identity + credential store on the PVC (the container's ~/.gitconfig and
# ~/.git-credentials vanish on every restart)
export GIT_CONFIG_GLOBAL=/workspace/.gitconfig
# These two are `if` blocks, not `[ -f x ] && source x`, and the file ends with
# a true command. env.sh is sourced by scripts that run under `set -e`, and a
# trailing && whose test fails makes SOURCING env.sh return 1 -- which aborts
# the caller. That is not hypothetical: it is what pod-build-stack.sh does on a
# fresh volume, where the venv it is about to create does not exist yet, so the
# build killed itself on its first run and left behind a log that just stopped.
# shellcheck disable=SC1091
if [ -f /workspace/gaussworks/scripts/gsplat-env.sh ]; then
    source /workspace/gaussworks/scripts/gsplat-env.sh
fi
# shellcheck disable=SC1091
if [ -f /workspace/venv/bin/activate ]; then
    source /workspace/venv/bin/activate
fi
:
