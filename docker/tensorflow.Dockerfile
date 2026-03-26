FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN python -m pip install --no-cache-dir \
    pyyaml \
    pillow

ENTRYPOINT ["python"]