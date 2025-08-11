"""
This script orchestrates the conversion of a Hugging Face model to the GGUF
format using the llama.cpp conversion script.

It first attempts to emit the requested precision directly via convert_hf_to_gguf.py.
If that fails (or is not supported for the requested precision), it falls back to:
  1) convert to f16 GGUF in a temporary directory, and
  2) quantize to the requested precision using the llama-quantize tool.

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
import shutil
import tempfile

# Define a global constant for the divider string used in logging
DIVIDER = '-------------------------------------------------------------'
# Path to the conversion script inside the Docker container
CONVERSION_SCRIPT_PATH = "/app/llama-cpp-python/vendor/llama.cpp/convert_hf_to_gguf.py"

def _run_and_stream(cmd):
    """Run a command, stream stdout->logger, and return the exit code."""
    logging.info(f"Executing command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8'
        )
        for line in iter(process.stdout.readline, ''):
            logging.info(line.strip())
        process.stdout.close()
        return process.wait()
    except Exception as e:
        logging.error(f"Subprocess exception: {e}")
        return 1

def _convert_to_gguf(input_model_dir, outfile, outtype):
    """
    Attempt a direct HF->GGUF conversion for the given outtype.
    Returns True on success, False otherwise.
    """
    if not os.path.exists(CONVERSION_SCRIPT_PATH):
        logging.error(f"Conversion script not found at: {CONVERSION_SCRIPT_PATH}")
        return False

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    cmd = [
        "python3",
        CONVERSION_SCRIPT_PATH,
        input_model_dir,
        "--outfile", outfile,
        "--outtype", outtype
    ]

    logging.info("Starting model conversion...")
    rc = _run_and_stream(cmd)
    if rc != 0:
        logging.warning(f"Direct conversion to '{outtype}' failed with return code {rc}.")
        return False

    logging.info("Direct conversion completed successfully.")
    return True

def _quantize_with_llama_quantize(infile_f16, outfile, quant_type):
    """
    Use llama-quantize to quantize an f16 GGUF into the requested quant_type.
    Returns True on success, False otherwise.
    """
    # Try to locate the binary; default to /usr/local/bin/llama-quantize
    quant_bin = shutil.which("llama-quantize") or "/usr/local/bin/llama-quantize"
    if not shutil.which(quant_bin) and not os.path.exists(quant_bin):
        logging.error("llama-quantize binary not found. Ensure it is installed in PATH or /usr/local/bin.")
        return False

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    cmd = [quant_bin, infile_f16, outfile, quant_type]
    logging.info(f"Quantizing '{infile_f16}' -> '{outfile}' as '{quant_type}' ...")
    rc = _run_and_stream(cmd)
    if rc != 0:
        logging.error(f"llama-quantize failed with return code {rc}.")
        return False

    logging.info("llama-quantize completed successfully.")
    return True

def converter(model_path, model_name, output_path, precision):
    """
    Converts a Hugging Face model to GGUF format.

    Args:
        model_path (str): The base directory where models are stored.
        model_name (str): The specific model directory to convert.
        output_path (str): The directory to save the output .gguf file.
        precision (str): The target precision for the conversion (e.g., 'f16', 'q8_0').

    Strategy:
      1) Try direct convert_hf_to_gguf.py to requested precision.
      2) If it fails and precision != f16, convert to f16 in a temp dir, then llama-quantize to requested precision.
    """
    input_model_dir = os.path.join(model_path, model_name)
    logging.info(f"Input model directory: {input_model_dir}")
    logging.info(f"Output directory: {output_path}")
    logging.info(f"Target precision: {precision}")

    if not os.path.isdir(input_model_dir):
        logging.error(f"Model directory not found: {input_model_dir}")
        sys.exit(1)

    # Final output filename in outputs/
    final_outfile = os.path.join(output_path, f"{model_name}.{precision}.gguf")

    # 1) Attempt direct conversion
    if _convert_to_gguf(input_model_dir, final_outfile, precision):
        return

    # 2) Fallback: generate f16 then quantize, only if requested precision isn't f16
    if precision == 'f16':
        logging.error("Requested precision is f16 and direct conversion failed; no quantization fallback possible.")
        sys.exit(1)

    logging.info("Falling back to: f16 conversion + llama-quantize pipeline.")

    temp_dir = tempfile.mkdtemp(prefix="gguf_tmp_")
    try:
        temp_f16 = os.path.join(temp_dir, f"{model_name}.f16.gguf")

        # 2.1) Convert to f16 in a temp directory
        if not _convert_to_gguf(input_model_dir, temp_f16, 'f16'):
            logging.error("Fallback f16 conversion failed; aborting.")
            sys.exit(1)

        # 2.2) Quantize temp f16 into the final requested precision
        if not _quantize_with_llama_quantize(temp_f16, final_outfile, precision):
            logging.error("llama-quantize step failed; aborting.")
            sys.exit(1)

        logging.info("Fallback pipeline (f16 -> quantize) completed successfully.")
    finally:
        # Clean up temp dir
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logging.info(f"Temporary directory removed: {temp_dir}")
        except Exception as e:
            logging.warning(f"Could not remove temporary directory '{temp_dir}': {e}")

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
        'Q8_0': 'q8_0',
        'Q6_K': 'q6_k',
        'Q5_K_M': 'q5_k_m',
        'Q5_K_S': 'q5_k_s',
        'Q5_0': 'q5_0',
        'Q5_1': 'q5_1',
        'Q4_K_M': 'q4_k_m',
        'Q4_K_S': 'q4_k_s',
        'Q4_0': 'q4_0',
        'Q4_1': 'q4_1',
        'Q3_K_M': 'q3_k_m',
        'Q3_K_S': 'q3_k_s',
        'Q3_K_L': 'q3_k_l',
        'Q2_K': 'q2_k',
        # Add more mappings as needed
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
