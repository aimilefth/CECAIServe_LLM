# GPT_2 Composer Flow

## Overview


This directory serves as the **input and configuration hub for the Composer module**. It contains all the necessary configuration files, model assets, additional libraries, and data required to build and package the GPT-2 model into runnable inference services (Artificial Intelligence Service Containers, or ASCIs) for various hardware and AccInf-Frameworks (AccInf-Platforms).

Each AccInf-Platform has its own subdirectory with specific configuration files to ensure optimal performance and compatibility. The Composer flow reads these files, along with the global configurations, to build the final Docker images.

## Directory Structure

```
.
├── AGX_ORIN_LLM_LLAMA
│   ├── composer_args_agx_orin_llm_llama.yaml
│   └── extra_pip_libraries_agx_orin_llm_llama.txt
├── AGX_ORIN_LLM_TRF
│   ├── composer_args_agx_orin_llm_trf.yaml
│   └── extra_pip_libraries_agx_orin_llm_trf.txt
├── AGX_ORIN_LLM_TRT
│   ├── composer_args_agx_orin_llm_trt.yaml
│   └── extra_pip_libraries_agx_orin_llm_trt.txt
├── AGX_ORIN_LLM_VLLM
│   ├── composer_args_agx_orin_llm_vllm.yaml
│   └── extra_pip_libraries_agx_orin_llm_vllm.txt
├── ARM_LLM_LLAMA
│   ├── composer_args_arm_llm_llama.yaml
│   └── extra_pip_libraries_arm_llm_llama.txt
├── ARM_LLM_TRF
│   ├── composer_args_arm_llm_trf.yaml
│   └── extra_pip_libraries_arm_llm_trf.txt
├── ARM_LLM_VLLM
│   ├── composer_args_arm_llm_vllm.yaml
│   └── extra_pip_libraries_arm_llm_vllm.txt
├── CPU_LLM_LLAMA
│   ├── composer_args_cpu_llm_llama.yaml
│   └── extra_pip_libraries_cpu_llm_llama.txt
├── CPU_LLM_TRF
│   ├── composer_args_cpu_llm_trf.yaml
│   └── extra_pip_libraries_cpu_llm_trf.txt
├── CPU_LLM_VLLM
│   ├── composer_args_cpu_llm_vllm.yaml
│   └── extra_pip_libraries_cpu_llm_vllm.txt
├── Client
│   ├── README.md
│   ├── composer_args_client.yaml
│   ├── extra_pip_libraries_client.txt
│   ├── my_client.py
│   └── prompts.json
├── GPU_LLM_LLAMA
│   ├── composer_args_gpu_llm_llama.yaml
│   └── extra_pip_libraries_gpu_llm_llama.txt
├── GPU_LLM_TRF
│   ├── composer_args_gpu_llm_trf.yaml
│   └── extra_pip_libraries_gpu_llm_trf.txt
├── GPU_LLM_TRT
│   ├── composer_args_gpu_llm_trt.yaml
│   └── extra_pip_libraries_gpu_llm_trt.txt
├── GPU_LLM_VLLM
│   ├── composer_args_gpu_llm_vllm.yaml
│   └── extra_pip_libraries_gpu_llm_vllm.txt
├── README.md
├── composer_args.yaml
├── dockerhub_config.yaml
├── experiment_server.py
└── extra_files_dir
```

## File Descriptions

### Global Configuration Files

- **`composer_args.yaml`**: Contains general arguments required for the Composer flow, applicable across all AccInf-Platforms (e.g., `APP_NAME_ARG`, `NETWORK_NAME_ARG`).
- **`dockerhub_config.yaml`**: Configuration for Docker Hub, specifying the repository and tag for the built ASCIs.
- **`experiment_server.py`**: Defines the `BaseExperimentServer` class. This is a crucial file that implements the **LLM-specific logic** for handling inference requests, such as decoding prompts from JSON, post-processing generated text, and encoding the final response.
- **`.env`** (optional, not shown): Can be used to define additional runtime environment variables that will be baked into the final Docker images.

### AccInf-Platform-Specific Configuration

Each AccInf-Platform directory (e.g., `GPU_LLM_TRF`, `CPU_LLM_LLAMA`) contains:
- **`composer_args_{AccInf-Platform}.yaml`**: AccInf-Platform-specific arguments passed during the Docker build. This includes critical parameters like `MODEL_NAME_ARG`, `BATCH_SIZE_ARG`, `PRECISION_ARG`, and generation knobs (`MAX_LENGTH_ARG`, `TEMPERATURE_ARG`).
- **`extra_pip_libraries_{AccInf-Platform}.txt`**: A list of any additional Python libraries to be installed in the final Docker image for that specific AccInf-Platform.

### Extra Files Directory

- **`extra_files_dir/`**: An optional directory where you can place any additional files (e.g., helper scripts, asset files) that need to be copied into the working directory of the final Docker images.


## Detailed Descriptions

### `composer_args.yaml`
This file sets global build arguments. For this project, it defines the application and network names used for metrics and logging.
```yaml
APP_NAME_ARG: Text_Generation
NETWORK_NAME_ARG: GPT2
SERVER_MODE_ARG: THR
```

### `dockerhub_config.yaml`
This file specifies where the built Docker images will be pushed. The `LABEL` is combined with the AccInf-Platform name to create the final image tag (e.g., `gpt2_gpu_llm_trf`).
```yaml
REPO: aimilefth/cecaiserve_llm
LABEL: gpt2
```

### `experiment_server.py`
This script defines the core logic for the LLM server. It inherits from `BaseServer` (located in `src/Composer`) and overrides methods to handle text-based I/O. Key methods include `decode_input` (parses incoming JSON with prompts), `create_and_preprocess` (batches the prompts), and `postprocess`/`encode_output` (formats the generated text into a JSON response).

### AGX Specific Files

#### `AGX/composer_args_agx.yaml`

Contains specific arguments for the AGX platform, including server IP, port, model name, batch size, and calibration file.

#### `AGX/extra_pip_libraries_agx.txt`

Lists additional Python libraries required for the AGX platform.

## Create Your Own Input Directory for the Composer Flow

This Composer directory serves as a template. Follow these instructions to customize it for your own model and needs:

### Step-by-Step Guide

1.  **Retain Necessary Subdirectories**:
    - Keep the subdirectories for the AccInf-Platforms you wish to build. You can safely remove any others.

2. **Adjust YAML Files**:
    - **Do not remove any `.yaml` files or change their field names.** Instead, adjust their values.
    - In `dockerhub_config.yaml`, update `REPO` and `LABEL` for your project.
    - In `composer_args_{pair}.yaml` files, configure the `MODEL_NAME_ARG`, `BATCH_SIZE_ARG`, `PRECISION_ARG`, and other parameters according to your model and performance targets.

3.  **Implement Custom Logic**:
    - If your experiment requires a different input/output format, modify the functions within **`experiment_server.py`**.
    - Ensure that any changes to the server's I/O logic are matched by corresponding changes in the client's logic in **`Client/my_client.py`**.

4.  **Extend Docker Container Functionality**:
    - **Additional Files**: Add any required files to the `extra_files_dir` directory.
    - **Environment Variables**: Define additional runtime environment variables in the `.env` file.
    - **Python Libraries**: Specify any extra Python libraries in the `extra_pip_libraries_{AccInf-Framework}.txt` files.

5.  **Prepare AI Model Files**:
    - The **CECAIServe** flow automatically places the correct model file (e.g., the original Hugging Face directory or the converted `.gguf` file) into the appropriate AccInf-Platform subdirectory.
    - If running the Composer flow standalone, you must **manually place the model assets** in the correct subdirectories (e.g., copy your `.gguf` file into `GPT_2/Composer/GPU_LLM_LLAMA/`).


## Conclusion

The Composer input directory is the central hub for configuring and building the final LLM inference services. By providing AccInf-Platform-specific configuration files, model assets, and custom logic, it allows the Composer to generate optimized and ready-to-deploy ASCIs. This documentation provides a comprehensive overview of the components within this directory, enabling you to customize the CECAIServe flow for your specific models and AccInf-Platforms.