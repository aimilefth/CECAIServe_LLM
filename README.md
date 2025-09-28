# CECAIServe_LLM: Cloud-Edge-Continuum Accelerated LLM Inference Serving

CECAIServe_LLM is a specialized version of the original **[CECAIServe](https://github.com/aimilefth/CECAIServe)** project, adapted and optimized specifically for deploying **Large Language Models (LLMs)**.

This repository builds upon the core two-stage workflow (Converter and Composer) of the original project but tailors every component for the unique demands of LLM inference serving. For a complete understanding of the foundational concepts like **Accelerated Inference Service Containers (ASCIs)**, **AccInf-Platforms**, and the general workflow, please refer to the **[README.md of the original CECAIServe repository](https://github.com/aimilefth/CECAIServe/blob/main/README.md)**.

### Additional Resources

For more details on the core components, refer to the documentation in the `src` subdirectories:
- [src/Base_container_images/README.md](src/Base_container_images/README.md)
- [src/Composer/README.md](src/Composer/README.md)
- [src/Converter/README.md](src/Converter/README.md)
- [src/CECAIServe_pipelines/README.md](src/CECAIServe_pipelines/README.md)