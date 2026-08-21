#!/usr/bin/env bash
# Derek Yuen
# 2026-08

#CUDA_VISIBLE_DEVICES=1
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
#TF_ENABLE_AUTO_MIXED_PRECISION=1 python3 ./train.py resnet34v3

#docker pull nvcr.io/nvidia/tensorflow:20.12-tf1-py3
#docker run -it  --gpus all --ipc=host --name ngc-tf115-cuda111 --hostname ngc-tf115-cuda111 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/app -w /app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash

#python3 ./train.py resnet34v3
TF_ENABLE_AUTO_MIXED_PRECISION=1 python3 ./train.py resnet34v3
