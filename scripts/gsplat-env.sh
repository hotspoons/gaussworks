# SPDX-License-Identifier: Apache-2.0
# Single source of truth for gsplat's JIT build environment. Source this from
# every entry point that imports gsplat.
#
# Why it must be shared: torch's cpp_extension hashes the build configuration
# (arch list + nvcc flags + source paths) into the cached extension. Launch the
# trainer with one flag set and the viewer with another and ninja rebuilds the
# whole extension from scratch — ~30 min per fused rasterizer kernel. Same env
# everywhere = compile once, ever.

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/workspace/.torch_extensions}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.pip-cache}

# Build only for the installed GPU. NOTE: the NVIDIA pytorch containers export
# TORCH_CUDA_ARCH_LIST with every supported arch ("7.5 8.0 ... 12.0+PTX"), so we
# override rather than defer. GAUSSWORKS_ARCH pins it by hand (mixed fleets).
if [ -n "${GAUSSWORKS_ARCH:-}" ]; then
    export TORCH_CUDA_ARCH_LIST="$GAUSSWORKS_ARCH"
else
    _arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$_arch" ] && export TORCH_CUDA_ARCH_LIST="$_arch"
fi

# cicc peaks ~9-20GB per job on the fused kernels; -O1 cuts that hard for a
# negligible runtime difference. Keep this identical across entry points.
export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:-"-Xcicc -O1"}

# Jobs sized off free RAM at ~10GB each, so a 32GB box degrades to 1 job.
if [ -z "${MAX_JOBS:-}" ]; then
    _avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
    _jobs=$(( _avail / 10 )); [ "$_jobs" -lt 1 ] && _jobs=1
    export MAX_JOBS=$_jobs
fi

echo "[gsplat-env] arch=${TORCH_CUDA_ARCH_LIST:-?} jobs=$MAX_JOBS flags='$NVCC_APPEND_FLAGS' cache=$TORCH_EXTENSIONS_DIR"
