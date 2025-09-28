FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

RUN pip install --no-cache-dir https://github.com/vllm-project/vllm/releases/download/v0.10.0/vllm-0.10.0+cu118-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

# Install opencv, requests and flask
RUN pip install --no-cache-dir opencv-python requests flask redistimeseries python-dotenv pynvml pyJoules