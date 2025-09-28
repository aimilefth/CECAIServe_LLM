#!/bin/bash
docker buildx build -f ./Dockerfile --platform linux/amd64 --tag aimilefth/cecaiserve_base_images:cpu_llm_trf --push .