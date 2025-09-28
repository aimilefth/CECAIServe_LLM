# Base container images


The `Base_container_images` directory provides a collection of Dockerfiles and build scripts to construct the foundational Docker images for various LLM-focused AccInf-Platforms.

These images are optimized for different frameworks on platforms such as NVIDIA AGX Orin, ARM CPUs, standard x86 CPUs, and server-grade NVIDIA GPUs. Each directory in this repository represents a specific AccInf-Platform (e.g., `GPU_LLM_TRT`) and contains two main files:

1.  **`Dockerfile`**: Contains all instructions for Docker to build the image, including the parent image and the necessary dependencies for the specific AccInf-Framework.
2.  **`docker_build.sh`**: A shell script that executes the `docker buildx build` command with the correct platform arguments to build and push the image to a container registry.

## Build Instructions

All images are pre-built and publicly available in the **[aimilefth/cecaiserve_base_images](https://hub.docker.com/r/aimilefth/cecaiserve_base_images)** Docker Hub repository. However, if you need to rebuild any of the base images, follow these instructions:

### Rebuilding a Base Container Image

1.  **Navigate to the appropriate directory**:
    Each sub-directory contains its own `Dockerfile` and build script. To build a specific image, navigate to its directory. For example:
    ```shell
    cd GPU_LLM_TRT
    ```
2.  **Modify the `docker_build.sh` script (Optional)**:
    To push to your own Docker Hub repository, change the image tag inside the `docker_build.sh` script. For example:
    ```bash
    # Original
    docker buildx build -f ./Dockerfile --platform linux/amd64 --tag aimilefth/cecaiserve_base_images:gpu_llm_trt --push .
    # Modified
    docker buildx build -f ./Dockerfile --platform linux/amd64 --tag your-repo/your-image:gpu_llm_trt --push .
    ```
3.  **Execute the build script**:
    Run the `docker_build.sh` script to build and push the image.
    ```shell
    bash docker_build.sh
    ```
4.  **Update Composer Dockerfiles**:
    If you change the name of a base image, remember to update the `FROM` directive in the corresponding `Dockerfile` within the `src/Composer/{AccInf-Platform}/` directory.

### Building All Images

You can use the `docker_build_all.sh` script to build all base images. It supports sequential (default) and parallel modes.
```shell
# Build sequentially
bash docker_build_all.sh

# Build in parallel
bash docker_build_all.sh -m parallel
```

## Hardware Platforms and Frameworks

### AGX Orin

These images target the **NVIDIA Jetson AGX Orin** platform (`linux/arm64`) and are based on NVIDIA's L4T JetPack containers.
-   **`AGX_ORIN_LLM_LLAMA`**: Builds `llama-cpp-python` with CUDA support for running GGUF models with GPU acceleration.
-   **`AGX_ORIN_LLM_TRF`**: Uses NVIDIA's PyTorch container to run standard Hugging Face `Transformers` models on the Orin GPU.
-   **`AGX_ORIN_LLM_TRT`**: Builds `TensorRT-LLM` from source, enabling highly optimized inference for supported models.
-   **`AGX_ORIN_LLM_VLLM`**: Installs a community-built `vLLM` wheel for high-throughput serving on Jetson devices.

### ARM

These images are for generic **ARMv8 (`aarch64`) CPUs** and are based on ARM's official PyTorch images or standard Python images.
-   **`ARM_LLM_LLAMA`**: Builds `llama-cpp-python` with OpenBLAS for efficient CPU-based GGUF model inference.
-   **`ARM_LLM_TRF`**: Uses ARM's PyTorch base image to run standard Hugging Face `Transformers` models.
-   **`ARM_LLM_VLLM`**: Builds `vLLM` from source for CPU-based execution.

### Client

The Docker image for the Client side is based on the official Python Docker image. This image includes Python and essential libraries required for the client side.

### CPU

These images are for standard **x86-64 CPUs** and are built on generic Python base images.
-   **`CPU_LLM_LLAMA`**: Builds `llama-cpp-python` with OpenBLAS for optimized CPU inference with GGUF models.
-   **`CPU_LLM_TRF`**: Installs `Transformers` and PyTorch (CPU version) for running standard Hugging Face models.
-   **`CPU_LLM_VLLM`**: Builds `vLLM` from source for high-performance CPU-based serving.

### GPU

These images target servers with **NVIDIA GPUs** (`linux/amd64`) and are based on official NVIDIA CUDA or PyTorch containers.
-   **`GPU_LLM_LLAMA`**: Builds `llama-cpp-python` with full CUDA support for GPU-offloaded GGUF inference.
-   **`GPU_LLM_TRF`**: Uses a standard PyTorch+CUDA image to run Hugging Face `Transformers` models with GPU acceleration.
-   **`GPU_LLM_TRT`**: Builds `TensorRT-LLM` from source to create highly optimized inference engines for NVIDIA server GPUs.
-   **`GPU_LLM_VLLM`**: Installs the official `vLLM` wheel for high-throughput, memory-efficient serving on NVIDIA GPUs.


## Tests

The following table summarizes the key software versions and target devices for each base image. All images have been successfully built and tested as part of the CECAIServe workflow.

| AccInf-Platform    | AccInf-Framework | Framework Version | Target Device    | Tested |
|--------------------|------------------|-------------------|------------------|--------|
| AGX_ORIN_LLM_LLAMA | `llama.cpp`      | `0.3.14`          | NVIDIA AGX Orin  | Yes    |
| AGX_ORIN_LLM_TRF   | `Transformers`   | `4.55.2`          | NVIDIA AGX Orin  | Yes    |
| AGX_ORIN_LLM_TRT   | `TensorRT-LLM`   | `0.18.0`          | NVIDIA AGX Orin  | Yes    |
| AGX_ORIN_LLM_VLLM  | `vLLM`           | `0.9.3`           | NVIDIA AGX Orin  | Yes    |
| ARM_LLM_LLAMA      | `llama.cpp`      | `0.3.16`          | Generic ARM      | Yes    |
| ARM_LLM_TRF        | `Transformers`   | `4.32.1`          | Generic ARM      | Yes    |
| ARM_LLM_VLLM       | `vLLM`           | `0.10.0`          | Generic ARM      | Yes    |
| CPU_LLM_LLAMA      | `llama.cpp`      | `0.3.14`          | Generic x86-64   | Yes    |
| CPU_LLM_TRF        | `Transformers`   | `4.54.1`          | Generic x86-64   | Yes    |
| CPU_LLM_VLLM       | `vLLM`           | `0.10.0`          | Generic x86-64   | Yes    |
| GPU_LLM_LLAMA      | `llama.cpp`      | `0.3.14`          | NVIDIA GPU (x86) | Yes    |
| GPU_LLM_TRF        | `Transformers`   | `4.54.1`          | NVIDIA GPU (x86) | Yes    |
| GPU_LLM_TRT        | `TensorRT-LLM`   | `0.14.0`          | NVIDIA GPU (x86) | Yes    |
| GPU_LLM_VLLM       | `vLLM`           | `0.10.0`          | NVIDIA GPU (x86) | Yes    |


## Docker Images

The Docker images built from these Dockerfiles are pushed to Docker Hub under the repository **`aimilefth/cecaiserve_base_images:<tag>`**, where `<tag>` corresponds to the AccInf-Platform name (e.g., `agx_orin_llm_trt`).