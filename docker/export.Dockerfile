FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY scripts/conda/patch_ultralytics_tflite_quant.py /tmp/patch_ultralytics_tflite_quant.py

RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3.10 -m pip install --no-cache-dir --upgrade ultralytics && \
    python3.10 /tmp/patch_ultralytics_tflite_quant.py

ENTRYPOINT ["python3.10"]
