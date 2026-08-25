
# Derek Y
# 2026-07
#
# contained environment for running LaneExtraction on Windows / WSL2
#

$ErrorActionPreference = "SilentlyContinue"

# install wsl2 first; reboot after install
wsl --install

# install docker desktop engine; reboot after install
winget install Docker.DockerDesktop


docker version


# may need to install docker desktop (windows) first (or an equivalent docker engine)
IMG="nvcr.io/nvidia/tensorflow:23.01-tf1-py3"

docker pull nvcr.io/nvidia/tensorflow:23.01-tf1-py3

#
docker run -it \
	--gpus all --ipc=host \
	--name tf115 --hostname tf115 \
	--ulimit memlock=-1 --ulimit stack=67108864 \
	-v "$(pwd)":/app -w /app \
	nvcr.io/nvidia/tensorflow:23.01-tf1-py3 bash

# for linux:
docker run -it \
	-u $(id -u):$(id -g) \
	--gpus all --ipc=host \
	--name tf115 --hostname tf115 \
	--ulimit memlock=-1 --ulimit stack=67108864 \
	-v "$(pwd)":/app -w /app \
	nvcr.io/nvidia/tensorflow:23.01-tf1-py3 bash

# missing within container:
pip3 install imageio
pip3 install tflearn==0.5.0
