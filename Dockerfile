# splat-pipeline: ai-dev-pod + SfM stack (COLMAP/GLOMAP, CUDA) + gsplat + ingest tools
#
# Licenses (see THIRD_PARTY.md): COLMAP BSD-3, GLOMAP BSD-3, gsplat Apache-2.0;
# ffmpeg/exiftool are invoked as subprocess tools (LGPL/GPL; Artistic/GPL).
#
# The base already carries torch/CUDA/NCCL and the devpod LWS launcher, so this
# image drops straight into a ZipspaceDeployment (deploy/zipspace.yaml).
# CUDA_ARCHS defaults cover the fleet: A100 (80), L40S (89), GH200/H200 (90),
# RTX Pro Blackwell (120). Trim per-arch via build args if build time hurts.

# A PUBLIC, MULTI-ARCH base, so this image can be built by CI and pulled by anything.
# It used to be the platform's ai-dev-pod, which needs Harbor credentials to pull and is
# not published for arm64 -- and the fleet this runs on (GH200) is arm64, so the image
# could never have been built for the machines that use it. nvidia/cuda ships nvcc for
# both architectures and nothing else is assumed: python, torch and the rest are
# installed explicitly below rather than inherited.
ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04

# --- SfM builder --------------------------------------------------------------
FROM ${BASE_IMAGE} AS sfm-builder
ARG COLMAP_VER=3.11.1
ARG GLOMAP_VER=1.0.0
ARG CUDA_ARCHS=80;89;90;120

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build build-essential git \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-test-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libflann-dev libsuitesparse-dev libcgal-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Ceres FROM SOURCE with CUDA. The distro libceres-dev has no CUDA support, so
# COLMAP's --Mapper.ba_use_gpu silently buys nothing when linked against it --
# and bundle adjustment inside the incremental mapper is the longest CPU stage
# in the whole pipeline. Building it here is what makes that flag real; poses.py
# probes colmap's linkage (cuSOLVER/cuSPARSE) before passing it.
ARG CERES_VER=2.2.0
# Ceres 2.2.0 OVERRIDES the architecture list. CMakeLists.txt line ~252 does a plain
#     set(CMAKE_CUDA_ARCHITECTURES "50;60;70;80")
# with no `if (NOT DEFINED ...)` and no cache, so -DCMAKE_CUDA_ARCHITECTURES is ignored
# and nvcc is handed compute_50 -- which CUDA 13 dropped:
#     nvcc fatal : Unsupported gpu architecture 'compute_50'
# There is no option to set instead, so the line is patched. The `grep -q` after it makes
# the patch ASSERT ITSELF: if a later Ceres writes that line differently the build fails
# here and loudly, rather than quietly compiling for architectures the fleet does not
# have. (scripts/pod-build-stack.sh never hit this -- it takes libceres-dev from apt,
# which has no CUDA at all. This is the path that actually gets GPU bundle adjustment.)
RUN git clone --depth 1 -b ${CERES_VER} https://github.com/ceres-solver/ceres-solver /tmp/ceres \
    && sed -i "s/set(CMAKE_CUDA_ARCHITECTURES \"50;60;70;80\")/set(CMAKE_CUDA_ARCHITECTURES \"${CUDA_ARCHS}\")/" /tmp/ceres/CMakeLists.txt \
    && grep -q "set(CMAKE_CUDA_ARCHITECTURES \"${CUDA_ARCHS}\")" /tmp/ceres/CMakeLists.txt \
    && cmake -S /tmp/ceres -B /tmp/ceres/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/sfm \
    && cmake --build /tmp/ceres/build --target install \
    && rm -rf /tmp/ceres

# Headless COLMAP (no Qt/GUI) with CUDA SIFT + CUDA bundle adjustment
RUN git clone --depth 1 -b ${COLMAP_VER} https://github.com/colmap/colmap /tmp/colmap \
    && cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF \
        -DCMAKE_PREFIX_PATH=/opt/sfm \
        -DCMAKE_INSTALL_PREFIX=/opt/sfm \
    && cmake --build /tmp/colmap/build --target install \
    && cp -r /tmp/colmap/pycolmap /tmp/pycolmap \
    && rm -rf /tmp/colmap

# GLOMAP: global SfM, much faster than incremental mapper on road sequences
# GLOMAP 1.0.0, not the latest: 1.2.0 vendors an Oct-2025 COLMAP whose rig
# model aborts when writing a reconstruction built from a COLMAP 3.11 database
# (see docs/LANDSCAPE.md). Two more traps in one line: GLOMAP's tags carry no
# "v" prefix, so `-b v1.0.0` fails the clone -- which is how this stage was
# silently absent from the running pod -- and its vendored COLMAP demands Qt5
# unless GUI_ENABLED is off.
RUN git clone --depth 1 -b ${GLOMAP_VER} https://github.com/colmap/glomap /tmp/glomap \
    && cmake -S /tmp/glomap -B /tmp/glomap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF \
        -DCMAKE_PREFIX_PATH=/opt/sfm \
        -DCMAKE_INSTALL_PREFIX=/opt/sfm \
    && cmake --build /tmp/glomap/build --target install \
    && rm -rf /tmp/glomap

# --- runtime ------------------------------------------------------------------
FROM ${BASE_IMAGE}
ARG GSPLAT_REF=main
ARG CUDA_ARCHS=80;89;90;120

USER root
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: frame extraction; exiftool: GPMF GPS from GoPro files -- but see the
# version note below, the packaged one is only here for its Perl dependencies.
# The -dev boost/ceres packages are the lazy way to satisfy the copied
# binaries' shared libs; fine for a dev image.
# NOTE: build-essential but deliberately NO cmake or ninja from apt. scikit-build-core
# brings its own CMake (4.4) and pybind11 needs it: with Ubuntu's 3.28 on PATH,
# scikit-build-core uses that instead and pybind11's python_add_library fails with
# "No SOURCES given to target: _core". Installing cmake here to be helpful is what
# caused that build. The -dev packages below are here because pycolmap is COMPILED in
# this stage: includes COLMAP's headers, which include Eigen, Boost, SQLite3 and the rest, so the
# runtime stage needs the same set the builder did. Discovered the expensive way --
# "Could NOT find SQLite3" after thirty minutes of Ceres, COLMAP and GLOMAP compiling
# perfectly. Adding them one per failed build costs half an hour each; this is the set
# that is already known to build COLMAP.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libimage-exiftool-perl curl ca-certificates git build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-test-dev libeigen3-dev libsqlite3-dev \
    libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev \
    libceres-dev libflann-dev libsuitesparse-dev libcgal-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# ExifTool from upstream, NOT the distro package. GoPro moved the GPMF GPS
# payload from GPS5 to GPS9 with the HERO11 generation, and the MAX 2 writes
# GPS9. An exiftool without GPS9 support parses the file, reports every other
# telemetry stream, and returns no GPS -- which downstream is indistinguishable
# from a camera that never got a fix, and quietly disables geo alignment,
# locality chunking and distance-based frame spacing. Ubuntu 24.04 ships 12.76.
ARG EXIFTOOL_VERSION=13.44
# From the GitHub TAG MIRROR, not exiftool.org. exiftool.org serves only the NEWEST
# release, so a pinned version 404s there the moment one is cut -- measured: 13.44 is
# already gone from exiftool.org and present on the mirror. scripts/pod-build-stack.sh
# has used the mirror for exactly this reason; this file had not caught up.
RUN mkdir -p /opt/exiftool \
    && curl -fsSL "https://github.com/exiftool/exiftool/archive/refs/tags/${EXIFTOOL_VERSION}.tar.gz" \
      | tar xz -C /opt/exiftool --strip-components=1 \
    && ln -sf /opt/exiftool/exiftool /usr/local/bin/exiftool \
    && exiftool -ver

COPY --from=sfm-builder /opt/sfm /opt/sfm
ENV PATH=/opt/sfm/bin:${PATH}

# gsplat from source (kernels for our arch list) + its examples, which provide
# the reference trainer that splatpipe train shells out to per chunk.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0+PTX"
# Python and torch EXPLICITLY. The old base carried them; this one does not, and
# inheriting an interpreter is how a build starts depending on something nobody
# declared. python3-dev is not optional: pycolmap is compiled below and CMake's
# FindPython needs the headers.
ARG TORCH_SPEC="torch==2.9.1 torchvision"
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && pip install --break-system-packages --no-cache-dir ${TORCH_SPEC} --index-url ${TORCH_INDEX}
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# pycolmap from the COLMAP we pinned, never from PyPI: there are NO aarch64 wheels at
# any version, and even on x86 the published wheel is a different COLMAP (4.2.0) from the
# binaries this image carries (3.11.1) -- and gsplat uses it to READ the reconstruction
# those binaries WROTE.
COPY --from=sfm-builder /tmp/pycolmap /tmp/pycolmap
RUN CMAKE_PREFIX_PATH=/opt/sfm \
    SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS};-DGUI_ENABLED=OFF;-DTESTS_ENABLED=OFF" \
    pip install --no-cache-dir /tmp/pycolmap && rm -rf /tmp/pycolmap

RUN git clone --recursive https://github.com/nerfstudio-project/gsplat /opt/gsplat \
    && cd /opt/gsplat && git checkout ${GSPLAT_REF} \
    && pip install --no-build-isolation . \
    && grep -vE "^pycolmap" examples/requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt
ENV GSPLAT_EXAMPLES=/opt/gsplat/examples

COPY . /opt/splatpipe
RUN pip install /opt/splatpipe

# Fail the BUILD, not a deployment three hours into a capture. Every one of these has
# been a real outage at some point today: exiftool without GPS9 reads as "the camera had
# no fix", a COLMAP without CUDA is 4x slower and says nothing, and a missing pycolmap
# only surfaces when the trainer tries to read a reconstruction.
RUN set -eux; \
    colmap -h 2>&1 | sed -n 2p | grep -q CUDA; \
    glomap -h >/dev/null; \
    [ "$(exiftool -listx | grep -c GPS9)" -gt 0 ]; \
    python -c "import torch, gsplat, pycolmap, cv2, splatpipe; print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'pycolmap', pycolmap.__version__)"; \
    splatpipe profiles | grep -q gopro-max2

USER 1000
