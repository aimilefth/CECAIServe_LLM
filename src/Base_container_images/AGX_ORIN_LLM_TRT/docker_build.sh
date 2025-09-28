#!/bin/bash

# Define the directory for temporary libraries
LIB_DIR="temp_jetson_libs"

echo "--- Preparing build context ---"

# Create the temporary directory and copy the required library
mkdir -p ${LIB_DIR}
cp /usr/lib/aarch64-linux-gnu/nvidia/* ./${LIB_DIR}/

echo "Copied usr/lib/aarch64-linux-gnu/nvidia  to ./${LIB_DIR}/"

# Run the Docker build command
echo "--- Starting Docker build ---"

docker build -f ./Dockerfile --tag aimilefth/cecaiserve_base_images:agx_orin_llm_trt --push . &> build.log

# Clean up the temporary directory
echo "--- Cleaning up build context ---"
rm -rf ${LIB_DIR}

echo "--- Build complete ---"