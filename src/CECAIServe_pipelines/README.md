# CECAIServe Pipelines

This directory contains the high-level orchestration scripts needed to run the full **CECAIServe** workflow for different AccInf-Platforms. Each subdirectory contains a script named `CECAIServe_${AccInf-Platform}.sh` that automates the Converter and Composer stages for the specific AccInf-Platform.


### `CECAIServe_all.sh`

The main script to run the entire flow for a defined set of AccInf-Platforms is `CECAIServe_all.sh`. This script can execute the individual `CECAIServe_{AccInf-Platform}.sh` scripts either serially or in parallel.

#### Usage

To run the flow, navigate to this directory and execute the `CECAIServe_all.sh` script:

```bash
bash CECAIServe_all.sh -p <relative_path_to_input_dir> [-m (serial | parallel)]
```

**Arguments:**

-   `-p`: The relative path from this directory to the input directory (e.g., `../../GPT_2`).
-   `-m`: (Optional) Specifies the execution mode. Default is `serial`.

**Example:**

To run the flow for the `GPT_2` example in parallel:

```bash
bash CECAIServe_all.sh -p ../../GPT_2 -m parallel
```

#### Configuration

The `CECAIServe_all.sh` script is configured by the `CECAIServe_args.yaml` file located in your input directory (e.g., `GPT_2/CECAIServe_args.yaml`).

**Example `CECAIServe_args.yaml`:**
```yaml
input_framework: TRF
run_converter: True
run_composer: True
compose_native_TRF: True
```

-   **`input_framework`**: Specifies the source model type. Currently, `TRF` (for Hugging Face Transformers) is supported.
-   **`run_converter`**: If `True`, the Converter stage is executed for supported pairs (e.g., `*_LLM_LLAMA`).
-   **`run_composer`**: If `True`, the Composer stage is executed to build the final ASCIs.
-   **`compose_native_TRF`**: If `True`, includes the native Transformers (`*_LLM_TRF`) AccInf-Platforms in the Composer process.

### Modifying AccInf-Platforms

To change which AccInf-Platforms are executed, modify the arrays (`dirs`, `native_trf_dirs`) inside the `CECAIServe_all.sh` script.

### Detailed Script Information

This script orchestrates the execution of individual `CECAIServe_{AccInf-Platform}.sh` scripts. It records the start time, parses input arguments, validates necessary files and variables, and then executes the scripts either serially or in parallel based on the mode specified.

Key Features:

- Supports serial and parallel execution modes.
- Logs output to a file in the logs directory.
- Validates the presence of required files and variables.
- Can extend the directories to process based on configuration options.

### Individual `CECAIServe_{pair}.sh` Scripts

Each AccInf-Platform has a corresponding `CECAIServe_{AccInf-Platform}.sh` script (e.g., `CECAIServe_gpu_llm_trt.sh`). These scripts handle the end-to-end workflow for a AccInf-Platform.

**Responsibilities:**
1.  Run the **Converter** stage if `run_converter` is `True` and the pair requires it (e.g., `GPU_LLM_LLAMA`).
2.  Move the converted model from the Converter's output directory to the Composer's input directory.
3.  Update the Composer's YAML configuration (`composer_args_{AccInf-Platform}.yaml`) with the correct model name and precision.
4.  Run the **Composer** stage if `run_composer` is `True`.
5.  Log all execution details and timings to the input directory's `logs` folder.
