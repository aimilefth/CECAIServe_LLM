FROM aimilefth/cecaiserve_base_images:cpu_llm_llama

RUN pip install --no-cache-dir transformers[torch]==4.55.0

# Install git to clone the llama.cpp repository
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Clone the llama.cpp repository which contains the conversion script
WORKDIR /app
RUN git clone \
      --branch v0.3.15 \
      --single-branch \
      --depth 1 \
      --recurse-submodules \
      --shallow-submodules \
      https://github.com/abetlen/llama-cpp-python.git

# Build the llama-quantize tool!
WORKDIR /app/llama-cpp-python/vendor/llama.cpp
# Build only the tools (CPU + OpenBLAS), make a Release build, and compile the quantizer
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BLAS=ON \
        -DLLAMA_BLAS_VENDOR=OpenBLAS \
    && cmake --build build --target llama-quantize -j "$(nproc)" \
    && install -Dm755 build/bin/llama-quantize /usr/local/bin/llama-quantize

RUN pip install --no-cache-dir sentencepiece

WORKDIR /app
# Arguments
ARG CONVERTER_APP_ARG=converter.py
ARG LOG_CONFIG_ARG=logconfig.ini
ARG SCRIPT_ARG=script.sh
ARG DEVICE_ARG=GPU_LLM_LLAMA
ARG MODELS_PATH_ARG=/models
ARG LOGS_PATH_ARG=/logs/${DEVICE_ARG}
ARG OUTPUTS_PATH_ARG=/outputs/${DEVICE_ARG}

# Environmental Variables
ENV CONVERTER_APP=${CONVERTER_APP_ARG}
ENV LOG_CONFIG=${LOG_CONFIG_ARG}
ENV SCRIPT=${SCRIPT_ARG}
ENV LOG_FILE=${LOGS_PATH_ARG}/Converter_${DEVICE_ARG}
ENV MODELS_PATH=${MODELS_PATH_ARG}
ENV LOGS_PATH=${LOGS_PATH_ARG}
ENV OUTPUTS_PATH=${OUTPUTS_PATH_ARG}

ENV MODEL_NAME=default_model

# Copy Files
COPY ${CONVERTER_APP_ARG} ${WORKING_DIR_ARG}
COPY ${LOG_CONFIG_ARG} ${WORKING_DIR_ARG}
COPY ${SCRIPT_ARG} ${WORKING_DIR_ARG}

CMD /bin/bash ${SCRIPT}