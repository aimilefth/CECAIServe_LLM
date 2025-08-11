"""
This script orchestrates the conversion of a Hugging Face model to the GGUF
format using the llama.cpp conversion script.

It is designed to be run within a Docker container where the llama.cpp
repository has been cloned and its dependencies are installed. The script
reads configuration from environment variables to determine the input model,
output path, and the desired quantization precision.

Main Features:
- Converts Hugging Face models to GGUF format.
- Supports configurable precision (e.g., FP16, Q8_0).
- Logs the conversion process and output for debugging.

Environment Variables:
- MODELS_PATH: Path to the directory containing the input model.
- MODEL_NAME: Name of the Hugging Face model directory to be converted.
- OUTPUTS_PATH: Path to the directory where the converted .gguf file will be saved.
- PRECISION: The desired output precision (e.g., 'FP16', 'Q8_0').
- LOG_CONFIG: Path to the logging configuration file.
"""

import os
import logging
import logging.config
import subprocess
import sys
import time

# Define a global constant for the divider string used in logging
DIVIDER = '-------------------------------------------------------------'
# Path to the conversion script inside the Docker container
CONVERSION_SCRIPT_PATH = "/app/llama-cpp-python/vendor/llama.cpp/convert_hf_to_gguf.py"

def converter(model_path, model_name, output_path, precision):
    """
    Converts a Hugging Face model to GGUF format using a subprocess call.

    Args:
        model_path (str): The base directory where models are stored.
        model_name (str): The specific model directory to convert.
        output_path (str): The directory to save the output .gguf file.
        precision (str): The target precision for the conversion (e.g., 'f16', 'q8_0').
    """
    input_model_dir = os.path.join(model_path, model_name)
    logging.info(f"Input model directory: {input_model_dir}")
    logging.info(f"Output directory: {output_path}")
    logging.info(f"Target precision: {precision}")

    if not os.path.isdir(input_model_dir):
        logging.error(f"Model directory not found: {input_model_dir}")
        sys.exit(1)

    if not os.path.exists(CONVERSION_SCRIPT_PATH):
        logging.error(f"Conversion script not found at: {CONVERSION_SCRIPT_PATH}")
        sys.exit(1)

    # Ensure output directory exists and build a concrete outfile path
    os.makedirs(output_path, exist_ok=True)
    outfile = os.path.join(output_path, f"{model_name}.{precision}.gguf")
    # The convert_hf_to_gguf.py script places the output file inside the specified
    # directory. The output filename is automatically generated based on the model name and precision.
    cmd = [
        "python3",
        CONVERSION_SCRIPT_PATH,
        input_model_dir,
        "--outfile",
        outfile,
        "--outtype",
        precision
    ]

    logging.info("Starting model conversion...")
    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

        # Stream the output in real-time
        for line in iter(process.stdout.readline, ''):
            logging.info(line.strip())

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            logging.error(f"Conversion process failed with return code {return_code}.")
            sys.exit(1)
        else:
            logging.info("Conversion process completed successfully.")

    except Exception as e:
        logging.error(f"An exception occurred during conversion: {e}")
        sys.exit(1)


def main():
    """Main function to configure and run the conversion process."""
    # Configure logging
    logging.config.fileConfig(os.environ['LOG_CONFIG'])
    # Parse required parameters from environment variables
    MODEL_PATH = os.environ['MODELS_PATH']
    MODEL_NAME = os.environ['MODEL_NAME']
    OUTPUT_PATH = os.environ['OUTPUTS_PATH']
    PRECISION_ARG = os.environ.get('PRECISION', 'FP16')  # Default to FP16 if not set

    # Map project-specific precision names to the script's expected format
    precision_map = {
        'FP32': 'f32',
        'FP16': 'f16',
        'Q8_0': 'q8_0'
        # Add other mappings as needed, e.g., Q4_0, Q4_K_M, etc.
    }
    # Default to f16 if the key is not in the map
    mapped_precision = precision_map.get(PRECISION_ARG.upper(), 'f16')


    # Log the parsed parameters
    logging.info(' Command line options:')
    logging.info(f'--model_path         : {MODEL_PATH}')
    logging.info(f'--model_name         : {MODEL_NAME}')
    logging.info(f'--output_path        : {OUTPUT_PATH}')
    logging.info(f'--precision (input)  : {PRECISION_ARG}')
    logging.info(f'--precision (mapped) : {mapped_precision}')
    logging.info(DIVIDER)

    # Record the start time
    global_start_time = time.perf_counter()
    
    # Execute the conversion
    converter(MODEL_PATH, MODEL_NAME, OUTPUT_PATH, mapped_precision)
    
    # Record the end time of the conversion
    global_end_time = time.perf_counter()
    
    # Log the total time taken for the conversion
    logging.info("Execution Time: %.3f" %(global_end_time - global_start_time))

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
