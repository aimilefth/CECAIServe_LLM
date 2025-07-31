#!/bin/bash
docker buildx build -f ./Dockerfile --platform linux/arm64 --tag aimilefth/cecaiserve_base_images:arm_llm_trf --push .