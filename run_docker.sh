#!/usr/bin/env bash
code_dir=`pwd`
docker run --rm -it --gpus all \
    -v $code_dir:/rice_classification \
    -p 8888:8888 \
    rice_classification:latest \
    bash -c "cd /rice_classification/ ; bash"