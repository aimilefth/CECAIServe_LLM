FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3.10-dev && \
    # Clean up apt caches to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    https://files.pythonhosted.org/packages/dc/82/7ab793eba3f4c3c84244e4c578ea838536d526db01c787448e8177404598/torch-2.3.0-cp310-cp310-manylinux2014_aarch64.whl \
    https://files.pythonhosted.org/packages/5e/0c/68d03a38f6ab2ba2b2829eb11b334610dd236e7926787f7656001b68e1f2/transformers-4.53.0-py3-none-any.whl \
    opencv-python \
    requests \
    flask \
    redistimeseries \
    python-dotenv
