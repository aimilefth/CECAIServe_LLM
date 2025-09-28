#!/bin/bash

# Function to update or add a YAML entry
update_yaml() {
    local yaml_file=$1
    local key=$2
    local value=$3

    if [ ! -f "$yaml_file" ]; then
        echo "YAML file $yaml_file does not exist."
    else
        if grep -q "^${key}:" "$yaml_file"; then
            sed -i "s/^${key}:.*$/${key}: ${value}/" "$yaml_file"
            echo "Updated ${key} in ${yaml_file} to ${value}"
        else
            # Ensure it adds a newline before appending if the file does not end with a newline
            sed -i -e '$a\' "$yaml_file"
            echo "${key}: ${value}" >> "$yaml_file"
            echo "Added ${key} to ${yaml_file}"
        fi
    fi
}

valid_hf_repo() {
    local d="$1"
    [[ -d "$d" ]] || return 1
    # need >=1 *.safetensors
    shopt -s nullglob
    local st=( "$d"/*.safetensors )
    shopt -u nullglob
    [[ ${#st[@]} -ge 1 ]] || return 1
    # need the three files below
    [[ -f "$d/config.json" ]]     || return 1
    [[ -f "$d/tokenizer.json" ]]  || return 1
    return 0
}

NAME=CPU_LLM_VLLM

# Record the start time (in seconds and nanoseconds)
start_time=$(date +%s%N)

# Check if the script receives exactly three arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <absolute_input_path> <run_converter> <run_composer> <SRC_DIR> <input_framework>"
    exit 1
fi

absolute_input_path=$1
run_converter=$2
run_composer=$3
SRC_DIR=$4
input_framework=$5

# Ensure the logs directory exists
mkdir -p "${absolute_input_path}"/logs

# Then, direct the output to the file in the logs directory
exec > >(tee -ai "${absolute_input_path}"/logs/CECAIServe_${NAME,,}.log)
exec 2>&1

# Print the current date and time
echo "CECAIServe_${NAME,,}.sh script started on: $(date -u +"%Y-%m-%d %H:%M:%S")"

# Move files from Converter to Composer and modify Composer Yamls
if [ "$run_converter" = "True" ] && [ "$run_composer" = "True" ]; then
    source_path=${absolute_input_path}/Converter/models/
    destination_path=${absolute_input_path}/Composer/${NAME}
    mkdir -p "$destination_path"

    # Scan immediate subdirectories in source_path for valid HF repos
    declare -a valid_dirs=()
    if [ -d "$source_path" ]; then
        while IFS= read -r -d '' d; do
            if valid_hf_repo "$d"; then
                valid_dirs+=("$d")
            fi
        done < <(find "$source_path" -mindepth 1 -maxdepth 1 -type d -print0)
    else
        echo "Source path $source_path does not exist."
    fi

    if (( ${#valid_dirs[@]} == 0 )); then
        echo "No valid Hugging Face repos found under $source_path"
    else
        if (( ${#valid_dirs[@]} > 1 )); then
            echo "Found multiple valid HF repos:"
            for d in "${valid_dirs[@]}"; do
                echo " - $d"
            done
            echo "Using the first one."
        fi

        src_dir="${valid_dirs[0]}"
        dir_name="$(basename "$src_dir")"

        echo "Copying HF repo dir: $src_dir -> $destination_path/"
        rm -rf "${destination_path}/${dir_name}"
        cp -a "$src_dir" "$destination_path/"

        composer_yaml_file="${destination_path}/composer_args_${NAME,,}.yaml"
        update_yaml "$composer_yaml_file" "MODEL_NAME_ARG" "$dir_name"
    fi
fi

# Run Composer
if [ "$run_composer" = "True" ]; then
    SRC_COMPOSER_DIR=${SRC_DIR}/Composer
    composer_path=$absolute_input_path/Composer
    echo "Started Composer for ${NAME}"
    # If the directory doesn't exist, print a message and dont run Composer
    if [ ! -d "${composer_path}/${NAME}" ]; then
        echo "Directory ${composer_path}/${NAME} does not exist."
        echo "Composer for ${NAME} will not run" 
    else
        bash "${SRC_COMPOSER_DIR}"/${NAME}/composer_${NAME,,}.sh "${composer_path}" "${SRC_COMPOSER_DIR}"
        echo "bash ${SRC_COMPOSER_DIR}/${NAME}/composer_${NAME,,}.sh ${composer_path} ${SRC_COMPOSER_DIR}"
        echo "Composer for ${NAME} ended"
    fi
fi

end_time=$(date +%s%N)
# Calculate the elapsed time in seconds with milliseconds
elapsed_time=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)
echo "CECAIServe_${NAME,,} execution time: $elapsed_time seconds"