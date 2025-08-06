# syntax=docker/dockerfile:1
#
# This Dockerfile builds a 'release' image for TensorRT-LLM, tailored for V100 GPUs.
#
# Key features:
# 1. Self-Contained: Clones the TensorRT-LLM repository internally.
# 2. Configurable Git Release: Use the GIT_RELEASE build-arg to specify a tag/branch.
# 3. V100 Compatibility: Uses the 25.01-py3 base image and sets correct component versions.
# 4. Correct CUDA Architectures: Defaults include '70-real' for V100.
# 5. Multi-Stage Build: Produces a clean, minimal release image.
#

# =================================================================================================
# Global Build Arguments (Available to all stages)
# =================================================================================================
# Argument to specify the Git release (tag or branch) to clone. Default: A recent tag known to work well.
ARG GIT_RELEASE=v0.15.0

# Argument to specify the target CUDA architectures, separated by semicolons.
# Defaults: 70-real (V100), 80-real (A100), 86-real (A10, etc.)
ARG CUDA_ARCHS="70-real;80-real;86-real"

# Base image tag for build. Using 25.01-py3 for V100 compatibility.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch
ARG BASE_TAG=24.10-py3

# =================================================================================================
# Stage 1: 'source' - Clone the repository from GitHub
# =================================================================================================
FROM alpine/git:latest AS source
# Pass the GIT_RELEASE argument into this stage
ARG GIT_RELEASE
# Install git-lfs for pulling model weights and other large files
RUN apk add --no-cache git-lfs && \
    git lfs install
WORKDIR /src
# Clone the specific release/branch of the repository, initialize submodules, and pull LFS files
RUN git clone --depth 1 --branch ${GIT_RELEASE} https://github.com/NVIDIA/TensorRT-LLM.git . && \
    git submodule update --init --recursive && \
    git lfs pull

# =================================================================================================
# Stage 2: 'devel' - The build environment with all dependencies
# =================================================================================================
FROM ${BASE_IMAGE}:${BASE_TAG} AS devel

# Add NVIDIA EULA and AI Terms labels
LABEL com.nvidia.eula="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/"
LABEL com.nvidia.ai-terms="https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/"

ENV BASH_ENV=${BASH_ENV:-/etc/bash.bashrc}
ENV ENV=${ENV:-/etc/shinit_v2}
SHELL ["/bin/bash", "-c"]

# Clean up the pip constraint file from the base NGC PyTorch image.
RUN [ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt || true

# Install base dependencies
COPY --from=source /src/docker/common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install CMake
COPY --from=source /src/docker/common/install_cmake.sh install_cmake.sh
RUN bash ./install_cmake.sh && rm install_cmake.sh

# Install ccache
COPY --from=source /src/docker/common/install_ccache.sh install_ccache.sh
RUN bash ./install_ccache.sh && rm install_ccache.sh

# Download & install TRT and other CUDA libs, passing the correct versions for 25.01
COPY --from=source /src/docker/common/install_tensorrt.sh install_tensorrt.sh
RUN bash ./install_tensorrt.sh && rm install_tensorrt.sh


# Install latest Polygraphy
# FIX: Patch the script to prevent build failure from 'pip cache purge'
COPY --from=source /src/docker/common/install_polygraphy.sh install_polygraphy.sh
RUN sed -i 's/pip3 cache purge/pip3 cache purge || true/g' install_polygraphy.sh && \
    bash ./install_polygraphy.sh && \
    rm install_polygraphy.sh

# Install mpi4py
COPY --from=source /src/docker/common/install_mpi4py.sh install_mpi4py.sh
RUN bash ./install_mpi4py.sh && rm install_mpi4py.sh

# PyTorch is already installed in the base image, so we skip re-installation.
COPY --from=source /src/docker/common/install_pytorch.sh install_pytorch.sh
RUN bash ./install_pytorch.sh skip && rm install_pytorch.sh

# =================================================================================================
# Stage 3: 'wheel' - Compile the TensorRT-LLM Python wheel
# =================================================================================================
FROM devel AS wheel
ARG CUDA_ARCHS # Inherit the global CUDA_ARCHS argument
WORKDIR /src/tensorrt_llm
# Copy the full source code into this stage
COPY --from=source /src/ .

RUN sed -i \
    's/raise RuntimeError("Volta architecture is deprecated support.")/print("[build_wheel]  WARNING: Volta (70-real) build; official support is deprecated.", file=sys.stderr)/' \
    scripts/build_wheel.py
# Create cache directories for pip and ccache to speed up builds
RUN mkdir -p /root/.cache/pip /root/.cache/ccache
ENV CCACHE_DIR=/root/.cache/ccache

# Build the TRT-LLM wheel using the specified CUDA architectures.
# The --benchmarks flag is included to build C++ benchmarks as well.
ARG BUILD_WHEEL_ARGS="--clean --benchmarks --cuda_architectures=${CUDA_ARCHS} --trt_root /usr/local/tensorrt"
ARG BUILD_WHEEL_SCRIPT="scripts/build_wheel.py"
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=${CCACHE_DIR} \
    python3 ${BUILD_WHEEL_SCRIPT} ${BUILD_WHEEL_ARGS}

# =================================================================================================
# Stage 4: 'release' - The final, production-ready image
# =================================================================================================
FROM devel AS release

# Set Git commit and TRT-LLM version as environment variables
# This is a robust way to make this metadata available to all processes in the container.
COPY --from=source /src/.git /tmp/src/.git
COPY --from=source /src/tensorrt_llm/version.py /tmp/src/tensorrt_llm/version.py
RUN cd /tmp/src && \
    GIT_COMMIT=$(git rev-parse HEAD) && \
    TRTLLM_VER=$(grep '__version__' tensorrt_llm/version.py | cut -d'"' -f2) && \
    echo "TRT_LLM_GIT_COMMIT=${GIT_COMMIT}" >> /etc/trt_llm_env && \
    echo "TRT_LLM_VERSION=${TRTLLM_VER}" >> /etc/trt_llm_env && \
    rm -rf /tmp/src
ENV TRT_LLM_GIT_COMMIT ""
ENV TRT_LLM_VERSION ""
# Source the file to set the ENV variables for the rest of the build and for the final image
RUN . /etc/trt_llm_env && \
    echo "export TRT_LLM_GIT_COMMIT=${TRT_LLM_GIT_COMMIT}" >> /etc/profile.d/trt_llm_env.sh && \
    echo "export TRT_LLM_VERSION=${TRT_LLM_VERSION}" >> /etc/profile.d/trt_llm_env.sh

WORKDIR /app/tensorrt_llm
# Copy the built wheel from the 'wheel' stage
COPY --from=wheel /src/tensorrt_llm/build/tensorrt_llm*.whl .

# Install the wheel and then remove it to keep the image clean
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorrt_llm*.whl && \
    rm tensorrt_llm*.whl && \
    pip cache purge || true

# Copy over examples, benchmarks, and other useful files from the source
ARG SRC_DIR=/src/tensorrt_llm
COPY --from=wheel ${SRC_DIR}/README.md ./
COPY --from=wheel ${SRC_DIR}/examples ./examples
COPY --from=wheel ${SRC_DIR}/benchmarks ./benchmarks
ARG CPP_BUILD_DIR=${SRC_DIR}/cpp/build
COPY --from=wheel \
     ${CPP_BUILD_DIR}/benchmarks/bertBenchmark \
     ${CPP_BUILD_DIR}/benchmarks/gptManagerBenchmark \
     benchmarks/cpp/

# Set up library paths and symlinks so the system can find the compiled libraries
RUN ln -sv $(python3 -c 'import site; print(f"{site.getsitepackages()[0]}/tensorrt_llm/libs")') lib && \
    test -f lib/libnvinfer_plugin_tensorrt_llm.so && \
    echo "/app/tensorrt_llm/lib" > /etc/ld.so.conf.d/tensorrt_llm.conf && \
    ldconfig

# Install opencv, requests and flask
RUN pip install --no-cache-dir "numpy<2" opencv-python requests flask redistimeseries python-dotenv pynvml pyJoules