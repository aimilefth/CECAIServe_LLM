#!/bin/bash
docker buildx build -f ./Dockerfile --platform linux/arm64 --tag aimilefth/cecaiserve_base_images:agx_orin_llm_vllm --push .