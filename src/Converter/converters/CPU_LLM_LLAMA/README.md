# converter_cpu_llm_llama.sh

## Configuration Files

Running the converter script requires the existence and correctness of two configuration files:
1. `converter_args.yaml`: Contains environmental variables that are applicable to all AI-framework/platform pairs.
2. `converter_args_cpu.yaml`: Contains environmental variables that are specific to the CPU_LLM_LLAMA AI-framework/platform pair.

**Pro Tip**: Ensure the .yaml files end with an empty line to avoid parsing issues.

## Example Configuration Files

### Example `converter_args.yaml`

```yaml
IMAGE_NAME: aimilefth/cecaiserve_llm_converter
MODEL_NAME: gpt2
TRAINED: True
MODELS_PATH: ../models
LOGS_PATH: ../logs
OUTPUTS_PATH: ../outputs


```

### Example `converter_args_cpu.yaml`

```yaml
*(empty)*

```

## Environmental Variables

### From `converter_args.yaml`

- **IMAGE_NAME**: The name of the Docker image used for the conversion.
- **MODEL_NAME**: The name of the model to be converted.
- **TRAINED**: Indicates whether the model is trained (`True`) or not (`False`).
- **MODELS_PATH**: The relative path to the directory containing the model.
- **LOGS_PATH**: The relative path to the directory where logs will be saved.
- **OUTPUTS_PATH**: The relative path to the directory where the output models will be saved.
