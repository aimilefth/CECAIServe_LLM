#!/bin/bash

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

NAME=CPU_LLM_LLAMA

# Record the Converter start time
start_time=$(date +%s%N)

# Check if the script receives exactly one argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <converter_path>"
    exit 1
fi

# Get the converter_path
converter_path=$1

# Ensure the logs and outputs directories exist
mkdir -p "${converter_path}/logs/${NAME}"
mkdir -p "${converter_path}/outputs/${NAME}"

# Then, Direct the output to the file in the logs directory
exec > >(tee -ai "${converter_path}"/logs/converter_${NAME,,}.log)
exec 2>&1

echo "Converter script started on: $(date -u +"%Y-%m-%d %H:%M:%S")"

converter_configs_path=${converter_path}/configurations
cd "${converter_configs_path}" || { echo "converter_${NAME,,}.sh Failure to cd ${converter_configs_path}"; exit 1; }

yaml_files=(
  "${NAME}/converter_args_${NAME,,}.yaml"
  "converter_args.yaml"
)
yaml_missing_files=()

# Check each file
for file in "${yaml_files[@]}"; do
  if [[ ! -e "$file" ]]; then
    yaml_missing_files+=("$file")
  fi
done

# If there are missing yaml files print them and exit
if [[ ${#yaml_missing_files[@]} -ne 0 ]]; then
  echo "The following required files are missing:"
  for missing in "${yaml_missing_files[@]}"; do
    echo "$missing"
  done
  exit 1
fi

# Get argument values
while IFS=": " read -r key value || [ -n "$key" ]; do
    # Remove any leading spaces on the value
    value=${value#"${value%%[![:space:]]*}"}
    # Export the key and value
    export "$key"="$value"
    # Print the key and value
    echo "$key: $value"
done < ./${NAME}/converter_args_${NAME,,}.yaml

# Get argument values
while IFS=": " read -r key value || [ -n "$key" ]; do
    # Remove any leading spaces on the value
    value=${value#"${value%%[![:space:]]*}"}
    # Export the key and value
    export "$key"="$value"
    # Print the key and value
    echo "$key: $value"
done < ./converter_args.yaml

# Get correct IMAGE_NAME
IMAGE_NAME=${IMAGE_NAME}:${NAME,,}

# Get absolute Paths (needed for docker run)
MODELS_PATH=/$(pwd)/${MODELS_PATH}
LOGS_PATH=/$(pwd)/${LOGS_PATH}
OUTPUTS_PATH=/$(pwd)/${OUTPUTS_PATH}

directories=(
  "$MODELS_PATH"
  "$LOGS_PATH"
  "$OUTPUTS_PATH"
  "${MODELS_PATH}/${MODEL_NAME}"
)
variables=(
)

missing_files=()
missing_directories=()
missing_variables=()

# Check each directory
for dir in "${directories[@]}"; do
  if [[ ! -d "$dir" ]]; then
    missing_directories+=("$dir")
  fi
done

# Check each file
for file in "${files[@]}"; do
  if [[ ! -e "$file" ]]; then
    missing_files+=("$file")
  fi
done

# Check each variable
for var in "${variables[@]}"; do
  if [[ -z "${!var}" ]]; then
    missing_variables+=("$var")
  fi
done

# If there are missing directories, files, or variables, print them and exit
if [[ ${#missing_directories[@]} -ne 0 ]] || [[ ${#missing_files[@]} -ne 0 ]] || [[ ${#missing_variables[@]} -ne 0 ]]; then
  if [[ ${#missing_directories[@]} -ne 0 ]]; then
    echo "The following required directories are missing:"
    for missing in "${missing_directories[@]}"; do
      echo "$missing"
    done
  fi

  if [[ ${#missing_files[@]} -ne 0 ]]; then
    echo "The following required files are missing:"
    for missing in "${missing_files[@]}"; do
      echo "$missing"
    done
  fi

  if [[ ${#missing_variables[@]} -ne 0 ]]; then
    echo "The following required variables are not set:"
    for missing in "${missing_variables[@]}"; do
      echo "$missing"
    done
  fi
  
  exit 1
fi

# Validate HF repo layout for the selected model
if ! valid_hf_repo "${MODELS_PATH}/${MODEL_NAME}"; then
  echo "Not a valid Hugging Face repo at: ${MODELS_PATH}/${MODEL_NAME}"
  echo "Expected: at least one *.safetensors and files: config.json, tokenizer.json, vocab.json"
  exit 1
fi

USER_ID=$(id -u)
GROUP_ID=$(id -g)

docker_run_params=$(cat <<-END
    -u ${USER_ID}:${GROUP_ID} \
    -v ${MODELS_PATH}:/models \
    -v ${LOGS_PATH}:/logs \
    -v ${OUTPUTS_PATH}:/outputs \
    --env MODEL_NAME=${MODEL_NAME} \
    --pull=always \
    --rm \
    --network=host \
    --name=converter_${NAME,,}_${MODEL_NAME} \
    ${IMAGE_NAME}
END
)

docker run \
  $docker_run_params

end_time=$(date +%s%N)
# Calculate the elapsed time in seconds with milliseconds
elapsed_time=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc)
echo "Converter ${NAME} execution time: $elapsed_time seconds"
