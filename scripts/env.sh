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
# shellcheck disable=SC1091
[ -f /workspace/gaussworks/scripts/gsplat-env.sh ] && source /workspace/gaussworks/scripts/gsplat-env.sh
# shellcheck disable=SC1091
[ -f /workspace/venv/bin/activate ] && source /workspace/venv/bin/activate
