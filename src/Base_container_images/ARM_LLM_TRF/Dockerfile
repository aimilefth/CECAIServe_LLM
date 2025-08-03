# Start from ARM's PyTorch Docker image
FROM armswdev/pytorch-arm-neoverse:r24.03-torch-2.2.1-onednn-acl

# Update and install necessary system dependencies
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends pkg-config=0.29.2-1ubuntu3 -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a single layer for efficiency.
# Pinning numpy<2.0 is the crucial fix to maintain compatibility
# with the libraries (PyTorch, SciPy) pre-installed in the base image.
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    "transformers[torch]" \
    opencv-python \
    requests \
    flask \
    redistimeseries \
    python-dotenv \