#!/bin/bash
docker buildx build -f ./Dockerfile --platform linux/amd64 --tag aimilefth/cecaiserve_base_images:gpu_llm_trt --push . &>build.log