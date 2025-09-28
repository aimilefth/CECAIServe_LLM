# converter_agx_orin_llm_llama.sh

## Overview
This component converts a Hugging Face model from the `.safetensors` format into the `.gguf` format, which is required by the `llama-cpp-python` framework running on an AGX Orin device.

## Configuration
The conversion process is configured through two YAML files:

1.  **`converter_args.yaml`**: Contains environmental variables that are applicable to all AccInf-Platfrom converters.
2.  **`converter_args_agx_orin_llm_llama.yaml`**: Contains variables specific to this AccInf-Platform.

### Key Environment Variables

#### From `converter_args.yaml`
-   **`IMAGE_NAME`**: The base name of the Docker image used for the conversion (e.g., `aimilefth/cecaiserve_llm_converter`). The script will append the platform name.
-   **`MODEL_NAME`**: The name of the model directory (e.g., `gpt2`) located under the `models` path.
-   **`MODELS_PATH`**: The relative path to the directory containing the input models.
-   **`OUTPUTS_PATH`**: The relative path where the converted `.gguf` model will be saved.

#### From `converter_args_agx_orin_llm_llama.yaml`
-   **`PRECISION`**: The target quantization type for the output `.gguf` model. This is a critical setting.
    -   Examples: `FP16`, `Q8_0`, `Q5_K_M`, `Q4_K_M`.

### Example Configuration

#### `converter_args.yaml`
```yaml
IMAGE_NAME: aimilefth/cecaiserve_llm_converter
MODEL_NAME: gpt2
MODELS_PATH: ../models
LOGS_PATH: ../logs
OUTPUTS_PATH: ../outputs
```

#### `converter_args_agx_orin_llm_llama.yaml`
```yaml
PRECISION: Q8_0
```

## Execution
The conversion is initiated by running the `converter_agx_orin_llm_llama.sh` script, which reads the configuration files and executes the conversion inside a Docker container.