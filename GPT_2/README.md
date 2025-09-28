# GPT_2

## Overview

This project demonstrates the full capabilities of the **CECAIServe** tool, encompassing both the **Converter** and **Composer** flows. It provides all the necessary data, code, and configuration files to run the CECAIServe flow and generate Large Language Model (LLM) inference servers for various hardware and AccInf-Framework combinations (AccInf-Platforms).

The project includes a Hugging Face Transformers model (`openai-community/gpt2`) within the `Converter/models` directory and a sample dataset of prompts within the `Composer/Client` directory. The `CECAIServe_all.sh` script initiates the full flow:
1.  The **Converter** flow is run for supported pairs (e.g., `*_LLM_LLAMA`) to convert the base model into the optimized GGUF format.
2.  The **Composer** flow takes the appropriate model (original or converted) and assembles a Docker-based inference service (an Artificial Intelligence Service Container, or ASCI) for each target AccInf-Platform.


## CECAIServe_args.yaml Configuration

The `CECAIServe_args.yaml` file contains high-level configurations that are read by the main orchestration script (`src/CECAIServe_pipelines/CECAIServe_all.sh`).

```yaml
input_framework: TRF
run_converter: True
run_composer: True
compose_native_TRF: True
```
- **`input_framework`**: Specifies the framework of the input model. `TRF` stands for Hugging Face Transformers.
- **`run_converter`**: If `True`, runs the Converter flow for AccInf-Platforms that require it (e.g., converting to GGUF for `llama-cpp-python`).
- **`run_composer`**: If `True`, runs the Composer flow to build the final ASCIs.
- **`compose_native_TRF`**: If `True`, includes native Transformers (`*_LLM_TRF`) AccInf-Platforms in the Composer process.

## Download Required Files

Before running the CECAIServe flow, you need to download the required GPT-2 model from Hugging Face using the provided `download_data.sh` script. This script downloads the repository and places it in the `Converter/models/gpt2` directory.

### Usage

To download the required files, navigate to this directory and execute the following command:

```bash
bash download_data.sh
```

## Create your own Input Directory

This directory serves as a template for running the CECAIServe flow for different AccInf-Platforms on a specific LLM. To customize the flow for your own model and needs, follow these steps:

### Step-by-Step Guide
1. **Adjust `CECAIServe_args.yaml` Values**:
    - Modify the values in the `CECAIServe_args.yaml` file to match your requirements. For example, you might choose to only run the Composer by setting `run_converter: False`.

2. **Follow the Converter README**:
    - Navigate to `Converter/README.md` for detailed instructions on setting up and running the Converter flow. This is essential if your target platform uses a framework like `llama-cpp-python` that requires a specific format like GGUF.

3. **Follow the Composer README**:
    - Navigate to `Composer/README.md` for detailed instructions on setting up and running the Composer flow. This includes configuring the platform-specific YAML files, adding additional libraries, and modifying server logic in `experiment_server.py`.

## Conclusion

The `GPT_2` directory provides a comprehensive template for leveraging the CECAIServe tool to convert and deploy LLMs across various hardware and software platforms. By following the detailed instructions in the README files and adjusting the configuration files to suit your specific model, you can efficiently set up and run the end-to-end flow. The provided GPT-2 example serves as a guide to help you understand and implement the process for other Hugging Face Transformers-based models.