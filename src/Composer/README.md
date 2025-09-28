# Composer Flow

## Overview

The Composer module is designed to facilitate the deployment and execution of AI models on various hardware platforms. Each platform has its specific server implementation, Dockerfile, and configuration scripts to ensure optimal performance and compatibility. This documentation covers the base functionality and specific implementations for the GPU_LLM_TRT platform, as an example AccInf-Platform.

### Directory Structure

```
.
├── GPU_LLM_TRT
│   ├── Dockerfile.gpu_llm_trt
│   ├── gpu_llm_trt_server.py
│   ├── composer_gpu_llm_trt.sh
│   ├── my_server.py
│   └── power_scraper.py
├── api_server.py
├── workflow_server.py
└── utils.py
```


### File Descriptions

#### Base Files

-   **`api_server.py`**: Manages a Flask web server that handles incoming API requests (`/api/infer`, `/api/metrics`, `/api/power`, `/api/logs`, `/api/shutdown`) and routes them to a worker thread for processing.
-   **`workflow_server.py`**: Provides the foundational server class (`WorkflowServer`) with core functionality for logging, metrics collection, and managing the inference workflow (decode, preprocess, experiment, postprocess, encode).
-   **`utils.py`**: Contains utility functions for Redis, metrics formatting, and other common tasks.

#### AccInf-Platform Directory (GPU_LLM_TRT)

-   **`Dockerfile.{AccInf-Platform}`**: Defines the Docker image for the specific AccInf-Platform (e.g., `Dockerfile.gpu_llm_trt`). It starts from a pre-built base image and copies the necessary server code, model, and configurations.
-   **`{AccInf-Platform}_server.py`**: Implements the AccInf-Platform-specific functionality (e.g., `GpuLlmTrtServer`). It inherits from `ModelServer` and implements the methods for loading a model and running inference using the  AccInf-Framework (e.g., `transformers`, `vLLM`, `TensorRT-LLM`).
-   **`my_server.py`**: A simple wrapper that instantiates the AccInf-Platform-specific server class. This file is imported by `api_server.py`.
-   **`power_scraper.py`**: Contains platform-specific code to measure power consumption.
-   **`composer_{AccInf-Platform}.sh`**: The main orchestration script for the AccInf-Platform. It reads configuration files, builds the Docker image with the correct arguments, and pushes it to a container registry.


### Usage

It is recommended to use the Composer as part of the full **CECAIServe** flow by executing the scripts in `src/CECAIServe_pipelines`. This ensures that models from the Converter step are correctly placed and configurations are synced.

However, the Composer can be run individually. To build the service for a specific pair, execute its composer script. For example, for the `GPU_LLM_TRT` platform:

```bash
# This script is typically called by the main CECAIServe pipeline script.
bash src/Composer/GPU_LLM_TRT/composer_gpu_llm_trt.sh /path/to/GPT_2/Composer /path/to/src/Composer
```

This script performs the following steps:

1.  **File and Variable Check**: Verifies that all required scripts and configuration (`.yaml`) files are present.
2.  **File Preparation**: Copies the necessary base server files and the pair-specific files into the build context.
3.  **Docker Image Build**: Builds the Docker image using the specified `Dockerfile.{AccInf-Platform}` and build arguments from the YAML files.
4.  **Docker Image Push**: Pushes the final image to the Docker repository defined in `dockerhub_config.yaml`.
5.  **Cleanup**: Removes temporary files from the build context.

Before running, ensure the input directory (e.g., `GPT_2/Composer/GPU_LLM_TRT/`) is correctly populated with the model files and the `composer_args_{AccInf-Platform}.yaml` and `extra_pip_libraries_{AccInf-Platform}.txt` files.

### Conclusion

The Composer flow is a robust and flexible framework for deploying and running AI models on a variety of AccInf-Platforms. By leveraging AccInf-Framework-specific configurations and Docker containers, it ensures that models are executed efficiently and effectively. The GPU_LLM_TRT platform serves as a prime example of how the Composer can be tailored to different hardware, providing a template for other AccInf-Platforms. This documentation provides a comprehensive overview of the core components and their roles, facilitating an understanding of the Composer flow and its customization for specific AccInf-Platforms.