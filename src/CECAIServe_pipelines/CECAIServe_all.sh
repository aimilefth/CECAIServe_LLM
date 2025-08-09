#!/bin/bash

# Record the start time (in seconds and nanoseconds)
start_time=$(date +%s%N)

# Array of directories to check
dirs=("CPU_LLM_VLLM" "Client")
native_trf_dirs=("CPU_LLM_TRF")

# Default values
input_path=""
mode="serial"

while getopts "p:m:" opt; do
  case $opt in
    p) input_path="$OPTARG";;
    m) mode="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Check if input path is provided
if [ -z "$input_path" ]; then
    echo "Usage: $0 -p <relative_path> [-m (serial | parallel)]"
    exit 1
fi


absolute_input_path="$(pwd)/${input_path}"

# Ensure the logs directory exists
mkdir -p "${absolute_input_path}"/logs

# Then, direct the output to the file in the logs directory
exec > >(tee -ai "${absolute_input_path}"/logs/CECAIServe_all.log)
exec 2>&1

# Print the current date and time
echo "CECAIServe_all.sh script started on: $(date -u +"%Y-%m-%d %H:%M:%S")"

# List of files to check
files=(
    "${absolute_input_path}/CECAIServe_args.yaml"
)
missing_files=()

# Check each file
for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

# If there are missing files, print them and exit
if [[ ${#missing_files[@]} -ne 0 ]]; then
    echo "The following required files are missing:"
    for missing in "${missing_files[@]}"; do
        echo "$missing"
    done
    exit 1
fi

# List of variables to check
variables=(
  "input_framework"
  "run_converter"
  "run_composer"
  "compose_native_TRF"
)

missing_variables=()

# Get argument values
while IFS=": " read -r key value || [ -n "$key" ]; do
    # Remove any leading spaces on the value
    value=${value#"${value%%[![:space:]]*}"}
    # Export the key and value
    export "$key"="$value"
    # Print the key and value
    echo "$key: $value"
done < "${absolute_input_path}"/CECAIServe_args.yaml

# Check each variable
for var in "${variables[@]}"; do
  if [[ -z "${!var}" ]]; then
    missing_variables+=("$var")
  fi
done

# If there are any missing variables, print them and exit
if [[ ${#missing_variables[@]} -ne 0 ]]; then
  echo "The following required variables are not set:"
  for missing in "${missing_variables[@]}"; do
    echo "$missing"
  done
  exit 1
fi


# Validate input_framework
if [[ "$input_framework" != "TRF" ]]; then
    echo "input_framework is ${input_framework}, not TRF (transformers!)"
    exit 1
fi

# Make sure true, yes Y works correctly
function strtobool () {
    case $(echo "$1" | tr '[:upper:]' '[:lower:]') in
        "true" | "yes" | "y" )
        echo "True"
        ;;
        *)
        echo "False"
        ;;
    esac
}

run_converter=$(strtobool "$run_converter")
run_composer=$(strtobool "$run_composer")
compose_native_TRF=$(strtobool "$compose_native_TRF")

echo "Running in $mode mode"

# If compose_native_TRF is True, then add the additional directories to the dirs array
if [[ "$compose_native_TRF" == "True" ]]; then
    dirs+=("${native_trf_dirs[@]}")
fi

# Declare an associative array to hold the PIDs of the child processes and their corresponding directory names
declare -A pids

# Iterate over the array and check each directory
for dir in "${dirs[@]}"; do
    # Construct the filename
    CECAIServe_filename="./$dir/CECAIServe_${dir,,}.sh"
    SRC_DIR="$(pwd)/.."
    # Check if the file exists in the directory
    if [ ! -f "$CECAIServe_filename" ]; then
        echo "The file $CECAIServe_filename doesn't exist."
        continue
    fi
    # If we get to this point, the CECAIServe exists, so we can run the script.
    if [ "$mode" == "parallel" ]; then
        # Run the child script in the background
        setsid bash ./"${dir}"/CECAIServe_"${dir,,}".sh "${absolute_input_path}" "${run_converter}" "${run_composer}" "${SRC_DIR}" "${input_framework}" >/dev/null 2>&1 < /dev/null &
        # Add the PID of the child process to the array with its corresponding directory name
        pids[$!]="$dir"
        echo "Process for directory $dir started with PID = $!."
    else
        # Run the child script in the foreground and wait for it to finish
        bash ./"${dir}"/CECAIServe_"${dir,,}".sh "${absolute_input_path}" "${run_converter}" "${run_composer}" "${SRC_DIR}" "${input_framework}"
        echo "bash ./${dir}/CECAIServe_${dir,,}.sh ${absolute_input_path} ${run_converter} ${run_composer} ${input_framework}"
        echo "Process for directory $dir finished."
    fi
done

# If in parallel mode, wait for all background processes to finish
if [ "$mode" == "parallel" ]; then
    # While there are still processes running
    while ((${#pids[@]})); do
        # For each PID in the list
        for pid in "${!pids[@]}"; do
            # If the process has finished
            if ! kill -0 "$pid" 2>/dev/null; then
                # Print a message with the directory name instead of the PID
                echo "Process for directory ${pids[$pid]} finished."
                # Remove the PID from the list
                unset "pids[$pid]"
            fi
        done
        # Sleep for a while to avoid excessive CPU usage
        sleep 1
    done
fi

echo "Finished CECAIServe runs of ${input_path}."

end_time=$(date +%s%N)
# Calculate the elapsed time in seconds with milliseconds
elapsed_time=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)
echo "CECAIServe_all execution time: $elapsed_time seconds"
