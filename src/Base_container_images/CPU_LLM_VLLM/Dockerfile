# Stage 1: Builder
# This stage uses the full python:3.12 image to compile the VLLM wheel.
# It has the necessary build tools and a pre-configured Python environment.
FROM python:3.12 AS builder

# Argument to specify which VLLM version (tag or branch) to build
ARG VLLM_VERSION=v0.10.0
# Support for building with non-AVX512 vLLM, default false: docker build --build-arg VLLM_CPU_DISABLE_AVX512="true" ...
ARG VLLM_CPU_DISABLE_AVX512=0
ENV VLLM_CPU_DISABLE_AVX512=${VLLM_CPU_DISABLE_AVX512}
# Support for building with AVX512BF16 ISA, default false: docker build --build-arg VLLM_CPU_AVX512BF16="true" ...
ARG VLLM_CPU_AVX512BF16=0
ENV VLLM_CPU_AVX512BF16=${VLLM_CPU_AVX512BF16}
# Support for building with AVX512VNNI ISA, default false: docker build --build-arg VLLM_CPU_AVX512VNNI="true" ...
ARG VLLM_CPU_AVX512VNNI=0
ENV VLLM_CPU_AVX512VNNI=${VLLM_CPU_AVX512VNNI}
# 1. Install system dependencies required for building VLLM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    gcc-12 \
    g++-12 \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Set gcc-12 as the default compiler for VLLM
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

WORKDIR /workspace

# 2. Clone the specified version of VLLM
RUN git clone --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git vllm_source
WORKDIR /workspace/vllm_source

# 3. Install Python build dependencies
# We install directly into the system python, as the container provides isolation.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 4. Build the VLLM wheel for CPU
ENV VLLM_TARGET_DEVICE=cpu
RUN python setup.py bdist_wheel


# Stage 2: Final Runtime Image
# This stage uses the python:3.12-slim image for a minimal footprint.
# It only contains the necessary runtime libraries and our final Python packages.
FROM python:3.12

# 1. Install runtime system dependencies
# libnuma1 is the runtime library for libnuma-dev.
# libtcmalloc is for performance. The rest are for opencv-python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnuma1 \
    libtcmalloc-minimal4 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the compiled wheel from the builder stage
COPY --from=builder /workspace/vllm_source/dist/*.whl /tmp/

RUN ls /tmp/

# 3. Install the VLLM wheel and your other required packages
# The --no-cache-dir flag helps keep the final image size down.
RUN pip install --no-cache-dir /tmp/*.whl \
    opencv-python \
    requests \
    flask \
    redistimeseries \
    python-dotenv \
    pyJoules \
    --extra-index-url https://download.pytorch.org/whl/cpu 

# 4. Set the LD_PRELOAD environment variable for performance (for x86)
# This is critical for preventing memory fragmentation issues with PyTorch.
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"