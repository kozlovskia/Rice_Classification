FROM nvcr.io/nvidia/pytorch:22.04-py3
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /rice_classification/
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt