# Utility Scripts

The `src/utils` directory contains helpful scripts for managing the workflow.

### `utils/docker_runs.sh`

This script is used to easily run a composed CECAIServe container on a host device with the correct `docker run` commands and arguments.

**Usage:**
```bash
bash utils/docker_runs.sh -n <repo_name> -a <app_name> -d <device> [-y <yaml_file>]
```
-   `-n`: The name of your Docker repository (e.g., `aimilefth`).
-   `-a`: The application label from your builds (e.g., `cecaiserve_llm`).
-   `-d`: The target AccInf-Platform (e.g., `GPU_LLM_TRT`, `AGX_ORIN_LLM_VLLM`).
-   `-y`: (Optional) Path to a YAML file containing runtime environment variables (e.g., to override `SERVER_PORT` or `BATCH_SIZE`).

**Example:**
```bash
bash utils/docker_runs.sh -n mydockerhub -a my-llm-app -d GPU_LLM_TRT -y ../../GPT_2/Composer/GPU_LLM_TRT/composer_args_gpu_llm_trt.yaml
```

### `utils/clean_input.sh`

This script cleans up logs, outputs, and (optionally) model files from an input directory to prepare it for a fresh run.

**Usage:**
```bash
bash utils/clean_input.sh <input_path> [clean_models]
```
-   `input_path`: The path to the input directory (e.g., `GPT_2`).
-   `clean_models`: (Optional) If set to `True`, it will remove copied/converted models from the `Composer` subdirectories, preserving only the configuration files.

**Example:**
```bash
# Clean logs and outputs only
bash utils/clean_input.sh ../../GPT_2

# Clean logs, outputs, and models
bash utils/clean_input.sh ../../GPT_2 True
```