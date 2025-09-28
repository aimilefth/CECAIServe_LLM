# CECAIServe Source Directory

## Overview

The CECAIServe project is designed to facilitate the conversion and deployment of Large Language Model (LLM) inference servers across various hardware platforms. The source directory (`src`) is structured to support both the conversion of models to optimized formats and the composition of containerized inference services, known as Artificial Intelligence Service Containers (ASCIs).

This directory contains the core logic, scripts, configuration files, and Dockerfiles to streamline the end-to-end process of creating and deploying high-performance LLM services.

## Directory Structure

```
.
├── Base_container_images
├── CECAIServe_pipelines
├── Composer
├── Converter
└── utils
```

### Base_container_images

This directory provides Dockerfiles and build scripts to create the foundational Docker images for different LLM-focused **AccInf-Platforms**. These images are optimized for frameworks like `Transformers`, `llama-cpp-python`, `vLLM`, and `TensorRT-LLM` across various hardware platforms, including NVIDIA AGX Orin, ARM CPUs, standard x86 CPUs, and server-grade NVIDIA GPUs.

**Files:**
- `Dockerfile`: Instructions to build the base image with all necessary dependencies.
- `docker_build.sh`: Script to build and push the Docker image to a container registry.

For detailed information, see the [Base_container_images/README.md](Base_container_images/README.md).

### CECAIServe_pipelines

This directory contains the high-level orchestration scripts to run the full CECAIServe workflow. It includes the main `CECAIServe_all.sh` script to run the entire flow for a defined set of AccInf-Platforms, and individual `CECAIServe_{AccInf-Platform}.sh` scripts that automate the Converter and Composer stages for each specific AccInf-Platform.

**Files:**
- `CECAIServe_all.sh`: Main script to run the entire flow for all specified AccInf-Platforms.
- Platform-specific subdirectories with their respective `CECAIServe_{AccInf-Platform}.sh` scripts.

For detailed information, see the [CECAIServe_pipelines/README.md](CECAIServe_pipelines/README.md).


### Composer

The Composer module is responsible for building and packaging AI models into runnable inference services (ASCIs). Each AccInf-Platform has a specific server implementation, Dockerfile, and configuration to ensure optimal performance and compatibility.

**Files:**
- `base_server.py`: Foundational server class with core functionality for logging, metrics, and managing the inference workflow.
- `flask_server.py`: Manages the Flask web server to handle incoming API requests.
- `utils.py`: Utility functions for Redis, metrics formatting, and other common tasks.
- AccInf-Platform-specific subdirectories (e.g., `GPU_LLM_TRT`) with their respective Dockerfiles and server implementations.

For detailed information, see the [Composer/README.md](Composer/README.md).
### Converter

The Converter module transforms input models into optimized formats required by specific LLM AccInf-Frameworks. The current implementation focuses on converting standard Hugging Face models into the **GGUF** format for use with `llama-cpp-python`.

**Directories:**
- `code`: Source code used to build the Docker images that perform the model conversions.
- `converters`: Scripts that orchestrate the conversion by running the appropriate Docker container.

For detailed information, see the [Converter/README.md](Converter/README.md).

## Usage

### Running the CECAIServe Flow

It is recommended to run the full CECAIServe flow by executing the `CECAIServe_pipelines/CECAIServe_all.sh` script. This script orchestrates both the Converter and Composer stages for the AccInf-Platforms specified within it.

#### Example Usage

To run the flow for the `GPT_2` example, navigate to the `src/CECAIServe_pipelines` directory and execute the following command:

```bash
bash CECAIServe_all.sh -p ../../GPT_2 -m parallel
```
*   The `-p` flag points to the relative path of your input directory (e.g., `GPT_2`).
*   The `-m` flag specifies parallel execution.

### Cleaning Up

Use the `utils/clean_input.sh` script to clean up logs, outputs, and other temporary files generated during a run.

#### Example Usage

Navigate to the `src/utils` directory and execute the following command:

```bash
# Clean logs and outputs, but keep models
bash clean_input.sh ../../GPT_2

# Clean everything, including models from Composer directories
bash clean_input.sh ../../GPT_2 True
```

## Conclusion

The `src` directory is the core of the CECAIServe project, providing all the necessary components for converting and deploying LLM inference services across a variety of hardware platforms and software frameworks. By following the detailed instructions and utilizing the provided scripts and configuration files, users can efficiently manage the entire workflow from model preparation to deployment.

For more detailed information, please refer to the README.md files in the respective subdirectories.

### Additional Resources

- [Base_container_images/README.md](Base_container_images/README.md)
- [CECAIServe_pipelines/README.md](CECAIServe_pipelines/README.md)
- [Composer/README.md](Composer/README.md)
- [Converter/README.md](Converter/README.md)
