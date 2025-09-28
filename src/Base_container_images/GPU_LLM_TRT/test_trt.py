import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pynvml
if not hasattr(pynvml, "__version__"):
    pynvml.__version__ = "12.0.0"  # any value >= '11.5.0' satisfies TRT-LLM's check

from tensorrt_llm import LLM, SamplingParams

def print_cuda_order_torch():
    import os, torch
    print("CUDA_DEVICE_ORDER =", os.getenv("CUDA_DEVICE_ORDER"))
    print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("NVIDIA_VISIBLE_DEVICES =", os.getenv("NVIDIA_VISIBLE_DEVICES"))
    n = torch.cuda.device_count()
    print(f"torch sees {n} GPU(s)")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"[torch] cuda:{i}  name={props.name}  cc={props.major}.{props.minor}  mem={props.total_memory/1024**3:.1f} GiB")



def main():
    # Model could accept HF model name, a path to local HF model,
    # or TensorRT Model Optimizer's quantized checkpoints like nvidia/Llama-3.1-8B-Instruct-FP8 on HF.

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )
    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The president of the United States is', Generated text: 'likely to nominate a new Supreme Court justice to fill the seat vacated by the death of Antonin Scalia. The Senate should vote to confirm the
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'
if __name__ == '__main__':
    print_cuda_order_torch()
    main()