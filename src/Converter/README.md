# Converter

This directory contains the source code and scripts for the **Converter** flow. Its primary purpose is to convert input models from a standard format into an optimized format required by specific LLM AccInf-Frameworks. The current implementation focuses on converting Hugging Face models into the GGUF format for use with llama-cpp-python.

### Supported Conversions

The Converter flow is currently specialized for the `LLAMA` framework, which utilizes `llama-cpp-python`.

-   **Input Format**: Standard Hugging Face model repository (containing `.safetensors` files, `config.json`, `tokenizer.json`, etc.).
-   **Output Format**: Quantized GGUF model (`.gguf`).
-   **Supported Pairs**:
    -   `AGX_ORIN_LLM_LLAMA`
    -   `ARM_LLM_LLAMA`
    -   `CPU_LLM_LLAMA`
    -   `GPU_LLM_LLAMA`
-   **Features**: The tool supports various quantization levels (e.g., `Q8_0`, `Q5_K_M`, `FP16`) which can be specified in the configuration files.

### Repository Structure

-   **`code` directory**: Contains subdirectories for each supported AI-framework/platform pair. This code is used to build the Docker images that perform the conversions.
    -   `converter.py`: The main Python script that handles the conversion logic. It uses `llama.cpp`'s tools to first convert the model to a base GGUF format (`FP16`) and then quantizes it to the desired `PRECISION`.
    -   `Dockerfile`: Defines the Docker environment for the conversion, including installing `llama-cpp-python` and its dependencies.
    -   `logconfig.ini`: Logging configuration.
    -   `README.md`: Documentation for the specific converter.
    -   `script.sh`: The entrypoint script executed inside the Docker container.
-   **`converters` directory**: Contains the shell scripts (`converter_{pair}.sh`) that orchestrate the conversion by running the appropriate Docker container with the correct volume mounts and environment variables.

### Input Directory Structure

For the Converter to function correctly, your main input directory (e.g., `GPT_2`) must contain a `Converter` directory with the following structure:

-   **`models`**: Contains the input Hugging Face model repository.
-   **`configurations`**: Contains the YAML configuration files that control the Converter flow.

**Example Tree of an `input_directory/Converter`:**

```
GPT_2/Converter/
├── configurations
│   ├── GPU_LLM_LLAMA
│   │   └── converter_args_gpu_llm_llama.yaml
│   ├── CPU_LLM_LLAMA
│   │   └── converter_args_cpu_llm_llama.yaml
│   │   ... (other pairs)
│   └── converter_args.yaml
└── models
    └── gpt2
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ... (other model files)
```

### Usage

While it is recommended to run the Converter as part of the full **CECAIServe** flow using the scripts in `src/CECAIServe_pipelines`, it can also be run individually.

To run the conversion for a specific pair, execute the corresponding script from the `converters` directory. For example, for `GPU_LLM_LLAMA`:

```bash
# This script is typically called by the main CECAIServe pipeline script.
bash src/Converter/converters/GPU_LLM_LLAMA/converter_gpu_llm_llama.sh /path/to/GPT_2/Converter
```

The script will:
1.  Read the configuration from the `.yaml` files.
2.  Pull the required Docker image (`aimilefth/cecaiserve_llm_converter:gpu_llm_llama`).
3.  Run the container, mounting the `models`, `logs`, and `outputs` directories.
4.  Execute the conversion process inside the container.
5.  The resulting `.gguf` file will be saved in `/path/to/GPT_2/Converter/outputs/GPU_LLM_LLAMA/`.
