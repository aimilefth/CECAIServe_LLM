FROM python:3.11

# Install the package
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*
    
RUN python -m pip install --no-cache-dir --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context huggingface-hub

# # do *not* let CMake inject -mcpu=native
# RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_ARM_NO_DOTPROD=ON -DGGML_ARM_F16=OFF" \
#     pip install --no-cache-dir llama_cpp_python --verbose
RUN pip install --no-cache-dir https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp311-cp311-linux_aarch64.whl
# Install opencv, requests and flask
RUN pip install --no-cache-dir opencv-python requests flask redistimeseries python-dotenv