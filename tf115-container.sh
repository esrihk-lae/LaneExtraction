#!/usr/bin/bash
#
#

docker run -it \
	--gpus all --ipc=host \
	--name tf115 --hostname tf115 \
	-v "$(pwd)":/app -w /app \
	nvcr.io/nvidia/tensorflow:23.01-tf1-py3 bash

