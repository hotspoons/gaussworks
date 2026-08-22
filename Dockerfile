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
    libeigen3-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libceres-dev libflann-dev libsuitesparse-dev libcgal-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Headless COLMAP (no Qt/GUI) with CUDA SIFT + CUDA bundle adjustment
RUN git clone --depth 1 -b ${COLMAP_VER} https://github.com/colmap/colmap /tmp/colmap \
    && cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/sfm \
    && cmake --build /tmp/colmap/build --target install \
    && rm -rf /tmp/colmap

# GLOMAP: global SfM, much faster than incremental mapper on road sequences
RUN git clone --depth 1 -b v${GLOMAP_VER} https://github.com/colmap/glomap /tmp/glomap \
    && cmake -S /tmp/glomap -B /tmp/glomap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_INSTALL_PREFIX=/opt/sfm \
    && cmake --build /tmp/glomap/build --target install \
    && rm -rf /tmp/glomap

# --- runtime ------------------------------------------------------------------
FROM ${BASE_IMAGE}
ARG GSPLAT_REF=main
ARG CUDA_ARCHS=80;89;90;120

USER root
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: frame extraction + v360; exiftool: GPMF GPS from GoPro files.
# The -dev boost/ceres packages are the lazy way to satisfy the copied
# binaries' shared libs; fine for a dev image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libimage-exiftool-perl \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev \
    libceres-dev libflann-dev libsuitesparse-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

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
