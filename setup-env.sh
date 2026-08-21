#!/usr/bin/bash
# Derek Y
# 2026-07
#
# NOT intended to be run directly
# For documentation purposes only. 




# for ubuntu 24.04 (tr520)
# system pre-requisites; requires root
if [ "$(id -u)" -ne 0 ]; then
    apt update && \
    apt install -y python3-pip python3-venv curl 
    
    # dependencies
    apt update && \
    apt install -y build-essential libssl-dev zlib1g-dev \
                   libbz2-dev libsqlite3-dev libffi-dev tk-dev

    sudo apt update && \
    sudo apt install -y libblas-dev liblapack-dev gfortran

fi


# - options:
#    1. docker
#    2. conda
#    3. pyenv
#    4. venv
#    5. miniconda



# docker: best image - 20.12-tf1-py3
#         tf1.15.2, CUDA 11.1.0, py3.8.5
# ref: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
docker pull nvcr.io/nvidia/tensorflow:20.12-tf1-py3
#docker run -it  --gpus all --ipc=host --name ngc-tf115-cuda111 --hostname ngc-tf115-cuda111 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/app -w /app nvcr.io/nvidia/tensorflow:20.12-tf1-py3 bash
# still need to install: pip3: imageio, tflearn==0.5.0, Pillow==8.4.0, 



# conda (anaconda)
# for ubuntu 24.04, if needed - only this version of anaconda will work
cd /tmp ; wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
#eval "$(conda shell.bash hook)"

conda config --set auto_activate_base false  # for post-install 




# pyenv
curl https://pyenv.run | bash

# ---

# venv
python3 -m venv tf115
source tf115/bin/activate

# add NVIDIA index:
pip config set global.extra-index-url https://pypi.ngc.nvidia.com



# maybe
pip install --upgrade setuptools wheel
pip3 install nvidia-pyindex



# requirements
pip3 install numpy==1.26.4              # required
pip3 install tensorflow-gpu==1.15.0
#pip3 install tensorflow-gpu==1.13.1    # testing

pip3 install scipy==1.1.0
pip3 install imageio

pip3 install "Pillow==8.4.0"
#pip3 install "Pillow==9.5.0"
pip3 install protobuf==3.13.0   # required
pip install "protobuf<=3.20.3"  # tensorflow-gpu 1.15.4 likes this one more  

#pip3 install opencv-python                         # experimental
pip3 install opencv-python==4.6.0.66                # if you need to annotate data
pip3 install opencv-python-headless==4.6.0.66       # if you don't need to annotate data; also good for docker ops
#pip3 install opencv-python==4.9.0.80

pip3 install tflearn==0.5.0
pip3 install tensorboard==1.15.0

pip3 install nvidia-cusolver-cu11	# solves some in-container libcuda* issues

pip3 install mapbox

# NOTE: skimage is a requirement for inferencing to generate graphs,
#       but skimage is not available via pip3 / python > 3.8
#       and installing scikit-image will replace numpy, scipy and pillow
pip3 install scikit-image==0.17.2       # tested aug 17 ok

#tested install scikit-image, then reinstall numpy==1.18.5 and scipy==1.1.0, Pillow8.4.0
# but it breaks other things ...


# if tflearn is needed:
#pip3 install git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing

# misc
pip3 install gdown



# --- 

# miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh
conda update conda
conda update --all
conda create --name TF1.15 python=3.8
conda activate TF1.15


pip3 install nvidia-tensorflow[horovod]
conda install -c conda-forge openmpi 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/TF1.15/lib/
mkdir tf-testcd tf-test
wget https://github.com/dbkinghorn/NGC-TF1-nvidia-examples/archive/main/NGC-TF1-nvidia-examples.tar.gz
tar xf NGC-TF1-nvidia-examples.tar.gz
cd NGC-TF1-nvidia-examples-main/cnn/
python resnet.py --layers=50 --batch_size=64 --precision=fp32


# You now have a tested and working TF 1.15 This is the latest build of what they are using on NGC nv21.08
# If you have already installed Anaconda Python instead of miniconda3 then just start at conda update conda and conda update --all
#  (That is important so that you are using a new enough version of pip to resolve the dependencies properly)


