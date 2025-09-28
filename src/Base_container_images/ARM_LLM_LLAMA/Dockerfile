FROM python:3.11

ARG ARM_ARCH=armv8.2-a
# Install the package
RUN apt update && apt install -y --no-install-recommends \
    libopenblas-dev=0.3.29+ds-3 \
    ninja-build=1.12.1-1+b1 \
    build-essential=12.12 \
    pkg-config=1.8.1-4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*
    
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
    huggingface-hub==0.34.4 \
    opencv-python==4.12.0.88 \
    requests==2.32.4 \
    flask==3.1.1 \
    redistimeseries==1.4.5 \
    python-dotenv==1.1.1

# do *not* let CMake inject -mcpu=native
RUN CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_ARM_ARCH=${ARM_ARCH} -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
    pip install \
    --no-cache-dir \
    --verbose \
    llama_cpp_python==0.3.16
# RUN pip install --no-cache-dir https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp311-cp311-linux_aarch64.whl