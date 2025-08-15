FROM python:3.11

# Install the package
RUN apt update && apt install -y libopenblas-dev ninja-build build-essential pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*
    
RUN python -m pip install --no-cache-dir --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context huggingface-hub

# do *not* let CMake inject -mcpu=native
RUN CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON" \
    pip install --no-cache-dir llama_cpp_python --verbose
# RUN pip install --no-cache-dir https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp311-cp311-linux_aarch64.whl
# Install opencv, requests and flask
RUN pip install --no-cache-dir opencv-python requests flask redistimeseries python-dotenv