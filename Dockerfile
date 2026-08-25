# splat-pipeline: ai-dev-pod + SfM stack (COLMAP/GLOMAP, CUDA) + gsplat + ingest tools
#
# Licenses (see THIRD_PARTY.md): COLMAP BSD-3, GLOMAP BSD-3, gsplat Apache-2.0;
# ffmpeg/exiftool are invoked as subprocess tools (LGPL/GPL; Artistic/GPL).
#
# The base already carries torch/CUDA/NCCL and the devpod LWS launcher, so this
# image drops straight into a ZipspaceDeployment (deploy/zipspace.yaml).
# CUDA_ARCHS defaults cover the fleet: A100 (80), L40S (89), GH200/H200 (90),
# RTX Pro Blackwell (120). Trim per-arch via build args if build time hurts.

ARG BASE_IMAGE=harbor.tools.basedweights.com/patapsco.ai/ai-dev-pod:0.12.0

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
RUN git clone --depth 1 -b ${CERES_VER} https://github.com/ceres-solver/ceres-solver /tmp/ceres \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libimage-exiftool-perl \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev \
    libceres-dev libflann-dev libsuitesparse-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# ExifTool from upstream, NOT the distro package. GoPro moved the GPMF GPS
# payload from GPS5 to GPS9 with the HERO11 generation, and the MAX 2 writes
# GPS9. An exiftool without GPS9 support parses the file, reports every other
# telemetry stream, and returns no GPS -- which downstream is indistinguishable
# from a camera that never got a fix, and quietly disables geo alignment,
# locality chunking and distance-based frame spacing. Ubuntu 24.04 ships 12.76.
ARG EXIFTOOL_VERSION=13.44
RUN curl -fsSL "https://exiftool.org/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz" \
      | tar xz -C /opt \
    && mv "/opt/Image-ExifTool-${EXIFTOOL_VERSION}" /opt/exiftool \
    && ln -sf /opt/exiftool/exiftool /usr/local/bin/exiftool \
    && exiftool -ver

COPY --from=sfm-builder /opt/sfm /opt/sfm
ENV PATH=/opt/sfm/bin:${PATH}

# gsplat from source (kernels for our arch list) + its examples, which provide
# the reference trainer that splatpipe train shells out to per chunk.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0+PTX"
RUN git clone --recursive https://github.com/nerfstudio-project/gsplat /opt/gsplat \
    && cd /opt/gsplat && git checkout ${GSPLAT_REF} \
    && pip install --no-build-isolation . \
    && pip install -r examples/requirements.txt
ENV GSPLAT_EXAMPLES=/opt/gsplat/examples

COPY . /opt/splatpipe
RUN pip install /opt/splatpipe

USER 1000
