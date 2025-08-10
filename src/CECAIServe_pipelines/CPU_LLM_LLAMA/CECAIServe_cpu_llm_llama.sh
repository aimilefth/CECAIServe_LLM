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
    [[ -f "$d/vocab.json" ]]      || return 1
    return 0
}

# Find first .gguf (depth 1..N)
find_first_gguf() {
    local root="$1"
    local first=""
    while IFS= read -r -d '' f; do first="$f"; break; done < <(find "$root" -type f -name "*.gguf" -print0 | sort -z)
    if [[ -n "$first" ]]; then
        echo "$first"
        return 0
    else
        return 1
    fi
}

NAME=CPU_LLM_LLAMA

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

# Move .gguf from Converter to Composer and patch YAML
if [ "$run_converter" = "True" ] && [ "$run_composer" = "True" ]; then
    source_path=${absolute_input_path}/Converter/outputs/${NAME}
    destination_path=${absolute_input_path}/Composer/${NAME}
    mkdir -p "$destination_path"
    # Move outputs to Composer folders
    # Arrays for extensions and corresponding YAML arguments
    extensions=( ".gguf")
    declare -A ext_to_arg=([".gguf"]="MODEL_NAME_ARG")
    declare -A moved_files
    # Process files for each extension
    for i in "${!extensions[@]}"; do
        ext="${extensions[i]}"
        mapfile -d '' files < <(find "${source_path}" -name "*$ext" -print0)
        if (( ${#files[@]} == 0 )); then
            echo "No files found in ${source_path} with extension $ext"
        elif (( ${#files[@]} > 1 )); then
            echo "More than one $ext file found in ${source_path}}:"
            for file in "${files[@]}"; do
                echo "$file"
            done
        else
            echo "Moving ${files[0]} to ${destination_path}"
            mv -f "${files[0]}" "${destination_path}"
            moved_files["$ext"]=$(basename "${files[0]}")
        fi
    done

    composer_yaml_file="${destination_path}/composer_args_${NAME,,}.yaml"  # Assuming a common YAML for simplicity
    # Updating YAML files based on the extension to argument mapping
    for ext in "${!moved_files[@]}"; do
        # Check if the extension has an ARG for the yaml file
        if [[ -n "${ext_to_arg[$ext]}" ]]; then
            file_name=${moved_files[$ext]}
            yaml_key=${ext_to_arg[$ext]}
            if [[ -n $file_name ]]; then
                update_yaml "$composer_yaml_file" "$yaml_key" "$file_name"
            else
                echo "No file name available, skipping YAML update."
            fi
        fi
    done
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