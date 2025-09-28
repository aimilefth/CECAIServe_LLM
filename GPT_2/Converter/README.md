# GPT_2 Converter Flow

## Overview

This directory contains the source code and scripts for the **Converter** flow. Its primary purpose is to convert input models from a standard format into an optimized format required by specific LLM AccInf-Frameworks. The current implementation focuses on converting Hugging Face models into the **GGUF** format for use with `llama-cpp-python`.

### Supported Conversions

The Converter flow is currently specialized for the `LLAMA` framework. Other frameworks like Transformers (`TRF`), vLLM (`VLLM`), and TensorRT-LLM (`TRT`) use the Hugging Face model format directly and do not require this conversion step.

-   **Input Format**: Standard Hugging Face model repository (containing `.safetensors` files, `config.json`, `tokenizer.json`, etc.).
-   **Output Format**: Quantized GGUF model (`.gguf`).
-   **Supported Pairs**:
    -   `AGX_ORIN_LLM_LLAMA`
    -   `ARM_LLM_LLAMA`
    -   `CPU_LLM_LLAMA`
    -   `GPU_LLM_LLAMA`
-   **Features**: The tool supports various quantization levels (e.g., `Q8_0`, `Q5_K_M`, `FP16`) which can be specified in the configuration files.

## Directory Structure

```
.
├── configurations
│   ├── AGX_ORIN_LLM_LLAMA
│   │   └── converter_args_agx_orin_llm_llama.yaml
│   ├── ARM_LLM_LLAMA
│   │   └── converter_args_arm_llm_llama.yaml
│   ├── CPU_LLM_LLAMA
│   │   └── converter_args_cpu_llm_llama.yaml
│   ├── GPU_LLM_LLAMA
│   │   └── converter_args_gpu_llm_llama.yaml
│   └── converter_args.yaml
└── models
    └── gpt2
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ... (other model files)
```

## File Descriptions

### Configurations Directory

- **`converter_args.yaml`**: Contains general arguments applicable to all converters, such as the input model name (`MODEL_NAME`) and paths.
- **`converter_args_{pair}.yaml`**: AccInf-Platform-specific arguments. For `LLAMA` pairs, the most important argument is **`PRECISION`**, which defines the target quantization type for the output GGUF file (e.g., `Q4_K_M`).

### Models Directory

- **`gpt2/`**: Contains the input Hugging Face model repository downloaded by the `download_data.sh` script.

## Create Your Own Input Directory for the Converter Flow

This Converter directory serves as a template for converting Hugging Face models to the GGUF format. Follow these instructions to customize it for your own model:

### Step-by-Step Guide

1. **Place Your Model**:
    - Add your Hugging Face model repository to the `models/` directory.

2. **Adjust Configuration YAML Files**:
    - **Do not remove `.yaml` files or change their field names.**
    - In `converter_args.yaml`, update `MODEL_NAME` to match the name of your model's directory.
    - In each `converter_args_{pair}.yaml` file, set the `PRECISION` to your desired quantization level. Common high-performance options include `Q8_0`, `Q5_K_M`, and `Q4_K_M`. `FP16` provides the highest quality with the largest file size.

## Usage

While it is recommended to run the Converter as part of the full **CECAIServe** flow using the scripts in `src/CECAIServe_pipelines`, it can also be run individually.

To run the conversion for a specific pair (e.g., `GPU_LLM_LLAMA`), the `CECAIServe` flow executes the corresponding script from `src/Converter/converters/`. For example:

```bash
# This script is typically called by the main CECAIServe pipeline script.
bash src/Converter/converters/GPU_LLM_LLAMA/converter_gpu_llm_llama.sh /path/to/GPT_2/Converter
```
The script will:
1.  Read the configuration from the `.yaml` files.
2.  Pull the required Docker image for conversion.
3.  Run the container, which executes the conversion logic.
4.  The resulting `.gguf` file will be saved in `/path/to/GPT_2/Converter/outputs/GPU_LLM_LLAMA/`.


## Conclusion

The Converter module is essential for preparing models for frameworks that require optimized formats like GGUF. By correctly configuring the YAML files and placing your model in the `models` directory, you can easily generate quantized models ready for efficient inference with `llama-cpp-python`.