#!/bin/bash

# Function to copy logs directories
copy_logs() {
    local source_path="$1"
    local target_path="$2"

    # Ensure the target path exists
    mkdir -p "$target_path"

    # List of directories to copy
    directories=("logs" "Converter/logs" "Composer/logs")

    for dir in "${directories[@]}"; do
        if [ -d "$source_path/$dir" ]; then
            # Create the directory structure in the target path
            mkdir -p "$target_path/$dir"

            # Copy the contents
            cp -r "$source_path/$dir/"* "$target_path/$dir/"
            echo "Copied $source_path/$dir to $target_path/$dir"
        else
            echo "Directory $source_path/$dir does not exist, skipping."
        fi
    done
}

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_path target_path"
    exit 1
fi

# Call the copy_logs function with provided arguments
copy_logs "$1" "$2"