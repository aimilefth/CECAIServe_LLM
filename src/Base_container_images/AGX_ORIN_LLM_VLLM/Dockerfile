# This Dockerfile builds a VLLM runtime image with CUDA support for NVIDIA Jetson AGX Orin
# by leveraging pre-built wheels from the Jetson AI Lab community PyPI.
# This approach is much faster as it avoids compiling VLLM from source.
#
# Target Platform: Jetson AGX Orin with JetPack 6 (L4T r36.x)

# Use the base image for JetPack 6 (L4T r36.x), which is based on Ubuntu 22.04
ARG L4T_VERSION=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_VERSION}

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies and Python
# The l4t-base:r36.3.0 image (Ubuntu 22.04) comes with Python 3.10 by default.
# We just need to install pip and some runtime libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10=3.10.12-1~22.04.10 \
        python3-pip=22.0.2+dfsg-1ubuntu0.6 \
        libnuma1=2.0.14-3ubuntu2 \
        ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
        libsm6=2:1.2.3-1build2 \
        libxext6=2:1.3.4-1build1 \
        libgl1=1.4.0-1 \
        libopenblas0-pthread=0.3.20+ds-1 \
        libgfortran5=12.3.0-1ubuntu1~22.04 \
    && rm -rf /var/lib/apt/lists/*


# 2. Set environment variables required by the Jetson AI Lab packages
# This is the FIX for potential hardcoded .dev domains inside the packages.
ARG JETSON_PYPI_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
ENV INDEX_HOST=jetson-ai-lab.io

# 3. Install the Jetson-specific packages from their dedicated index.
# We use --index-url and NO --extra-index-url to FORCE pip to use only this source.
# This prevents it from downloading the wrong, generic torch wheel from pypi.org.
RUN pip install --no-cache-dir \
    --index-url ${JETSON_PYPI_INDEX} \
    "numpy==1.26.4" \
    "torch==2.8.0" \
    "torchvision==0.23.0" \
    "triton==3.4.0" \
    #"xformers==0.0.32+8ed0992.d20250724" \ # Install this to add FP32 support!
    "vllm==0.9.3+cu126" 

# 4. Install the remaining generic Python packages from the standard PyPI.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python==4.11.0.86 \
    requests==2.32.4 \
    flask==3.1.1 \
    redistimeseries==1.4.5 \
    python-dotenv==1.1.1
