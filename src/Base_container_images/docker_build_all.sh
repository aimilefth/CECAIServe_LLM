#!/bin/bash

# Array of subdirectories
# AGX_ORIN_LLM_TRT Requires building on AGX Orin!
subdirs=(AGX_ORIN_LLM_LLAMA AGX_ORIN_LLM_VLLM ARM_LLM_LLAMA ARM_LLM_VLLM CPU_LLM_LLAMA CPU_LLM_VLLM GPU_LLM_LLAMA GPU_LLM_TRT GPU_LLM_VLLM AGX_ORIN_LLM_TRF ARM_LLM_TRF CPU_LLM_TRF GPU_LLM_TRF)

# Default mode
mode="serial"

# Process command line options
while getopts "m:" opt; do
  case $opt in
    m) mode="$OPTARG";;  # Mode can be 'serial' or 'parallel'
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

echo "Running in $mode mode."

# Declare an associative array to hold the PIDs of the child processes and their corresponding directory names if in parallel mode
declare -A pids

# Iterate through each subdirectory and run docker_build.sh according to the selected mode
for subdir in "${subdirs[@]}"; do
    if [ -d "$subdir" ] && [ -f "$subdir/docker_build.sh" ]; then
        echo "Building in directory: $subdir"
        cd "$subdir"
        if [ "$mode" == "parallel" ]; then
            # Run the docker_build.sh script in the background
            setsid bash docker_build.sh >/dev/null 2>&1 < /dev/null &
            # Add the PID of the child process to the array with its corresponding directory name
            pids[$!]="$subdir"
            echo "Process for directory $subdir started with PID = $!."
        else
            # Run the docker_build.sh script in the foreground
            bash docker_build.sh
            echo "Process for directory $subsidir finished."
        fi
        cd - > /dev/null
    else
        echo "Skipping directory: $subdir (either directory or docker_build.sh does not exist)"
    fi
done

# If in parallel mode, wait for all background processes to finish
if [ "$mode" == "parallel" ]; then
    while ((${#pids[@]})); do
        for pid in "${!pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Process for directory ${pids[$pid]} finished."
                unset "pids[$pid]"
            fi
        done
        sleep 1
    done
fi

echo "Finished all docker_build.sh runs."