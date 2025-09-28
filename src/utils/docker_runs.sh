#!/bin/bash

# ADDED: Boolean flags for interactive mode and container naming
INTERACTIVE=True
CONTAINER_NAME=False

# Default values
image_name=""
image_app=""
device=""
yaml_file=""
supported_devices="AGX_ORIN_LLM_LLAMA AGX_ORIN_LLM_TRT AGX_ORIN_LLM_VLLM ARM_LLM_LLAMA ARM_LLM_VLLM CPU_LLM_LLAMA CPU_LLM_VLLM CLIENT GPU_LLM_LLAMA GPU_LLM_TRT GPU_LLM_VLLM AGX_ORIN_LLM_TRF ARM_LLM_TRF CPU_LLM_TRF GPU_LLM_TRF"

# Help message for usage
usage() {
    echo "Usage: $0 -n <image_name> -a <image_app> -d <device> [-y <yaml_file>]"
    echo "Devices: $(echo $supported_devices | tr ' ' ', ')"
    exit 1
}

# Process command-line options
while getopts "n:a:d:y:" opt; do
  case $opt in
    n) image_name="$OPTARG" ;;
    a) image_app="$OPTARG" ;;
    d) device="$OPTARG" ;;
    y) yaml_file="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Validate inputs
if [[ -z "$image_name" || -z "$image_app" || -z "$device" ]]; then
    echo "Options -n, -a, and -d must be provided."
    usage
fi

# Normalize device input to uppercase for comparison
device_upper_case=$(echo "$device" | tr '[:lower:]' '[:upper:]')
device_lower_case=${device_upper_case,,}

# Check if the device type is supported
if [[ ! " $supported_devices " =~ " $device_upper_case " ]]; then
    echo "Error: Unsupported device type '$device'."
    usage
fi

# Construct the image tag
IMAGE="${image_name}:${image_app}_${device_lower_case}"

# Check if the Docker image can be pulled
if ! docker pull "$IMAGE"; then
    echo "Error: Image '$IMAGE' cannot be pulled from the remote repository."
    exit 1
fi

env_vars=""
if [[ -n "$yaml_file" ]]; then
    # Check if the YAML file exists
    if [ ! -f "$yaml_file" ]; then
        echo "Error: YAML file '$yaml_file' not found."
        exit 1
    fi

    # Read and parse the YAML file into Docker environment variables
    echo "YAML Environment Variables:"
    while IFS=": " read -r key value || [ -n "$key" ]; do
        value=${value#"${value%%[![:space:]]*}"} # Remove leading spaces
        env_vars+="--env $key='$value' "
        export "$key"="$value"
        echo "$key=$value"
    done < "$yaml_file"
fi

# ADDED: Choose the correct interactive flags
if [ "$INTERACTIVE" = "True" ]; then
    interactive_flag="-it"
else
    interactive_flag="-t"
fi

# Start forming the docker run command with the chosen interactive option
docker_run_command="docker run $interactive_flag --rm --network=host --pull=always --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $env_vars -v /scrape:/scrape "

case "$device_upper_case" in
    AGX_ORIN_LLM_LLAMA|AGX_ORIN_LLM_TRT|AGX_ORIN_LLM_VLLM|AGX_ORIN_LLM_TRF)
        docker_run_command+="--runtime=nvidia "
        ;;
    GPU_LLM_LLAMA|GPU_LLM_TRT|GPU_LLM_VLLM|GPU_LLM_TRF)
        # Add CPU limits if NUM_CPUS is set
        if [[ -n "$NUM_CPUS" ]]; then
            docker_run_command+="--cpus=${NUM_CPUS} "
        fi
        # Add CPU set if CPU_SET is set
        if [[ -n "$CPU_SET" ]]; then
            docker_run_command+="--cpuset-cpus=${CPU_SET} "
        fi
        docker_run_command+="--gpus all --privileged "
        ;;
    CPU_LLM_LLAMA|CPU_LLM_VLLM|CPU_LLM_TRF)
        # Add CPU limits if NUM_CPUS is set
        if [[ -n "$NUM_CPUS" ]]; then
            docker_run_command+="--cpus=${NUM_CPUS} "
        fi
        # Add CPU set if CPU_SET is set
        if [[ -n "$CPU_SET" ]]; then
            docker_run_command+="--cpuset-cpus=${CPU_SET} "
        fi
        docker_run_command+="--privileged "
        ;;
    CLIENT)
        if [[ -n "$HOST_MOUNTED_DIR" ]]; then
            docker_run_command+="-v $HOST_MOUNTED_DIR:/home/Documents/mounted_dir "
        fi
esac

# ADDED: Conditionally add the --name argument
if [ "$CONTAINER_NAME" = "True" ]; then
    docker_run_command+="--name ${image_app}_${device} "
fi

docker_run_command+="$IMAGE"

# Add COMMAND as the final argument if it is set
if [[ -n "$COMMAND" ]]; then
    docker_run_command+=" $COMMAND"
fi

# Display the complete Docker run command before executing it
echo "Complete Docker run command:"
echo "$docker_run_command"

# Execute the Docker command
eval "$docker_run_command"
