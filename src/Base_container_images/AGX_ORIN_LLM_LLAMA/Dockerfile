FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

RUN apt-get update -y && apt-get install -y \
    git=1:2.34.1-1ubuntu1.15 \
    build-essential=12.9ubuntu3 \
    python3=3.10.6-1~22.04.1 \
    python3-pip=22.0.2+dfsg-1ubuntu0.6 \
    gcc=4:11.2.0-1ubuntu1 \
    wget=1.21.2-2ubuntu1.1 \
    ocl-icd-opencl-dev=2.2.14-3 \
    opencl-headers=3.0~2022.01.04-1 \
    clinfo=3.0.21.02.21-1 \
    libclblast-dev=1.5.2-2 \
    libopenblas-dev=0.3.20+ds-1 \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# setting build related env vars
ENV GGML_CUDA=1

RUN pip install \
    --no-cache-dir \
    --upgrade pip==25.2 \
    pytest==8.4.1 \
    cmake==4.1.0 \
    scikit-build==0.18.1 \
    setuptools==80.9.0 \
    fastapi==0.116.1 \
    uvicorn==0.35.0 \
    sse-starlette==3.0.2 \
    pydantic-settings==2.10.1 \
    starlette-context==0.4.0 \
    huggingface-hub==4.12.0.88 \
    opencv-python==4.12.0.88 \
    requests==2.32.4 \
    flask==3.1.1 \
    redistimeseries==1.4.5 \
    python-dotenv==1.1.1

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" pip install llama-cpp-python==0.3.14