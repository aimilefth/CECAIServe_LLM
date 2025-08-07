# Start from ARM's PyTorch Docker image
FROM armswdev/pytorch-arm-neoverse:r24.07-torch-2.3.0-onednn-acl

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
    numpy==1.25.2 \
    transformers[torch]==4.32.1 \
    opencv-python==4.11.0.86 \
    requests==2.31.0 \
    flask==3.1.1 \
    redistimeseries==1.4.5 \
    python-dotenv==1.1.1