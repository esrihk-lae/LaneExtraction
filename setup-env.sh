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
fi


# venv
python3 -m venv tf115
source tf115/bin/activate



# requirements
#pip3 install tensorflow-gpu==1.13.1     # testing
pip3 install tensorflow==1.15.0
pip3 install "Pillow==9.5.0"
pip3 install opencv-python
pip3 install numpy==1.26.4
pip3 install opencv-python==4.9.0.80


# if tflearn is needed:
#pip3 install git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing


# misc
pip3 install gdown
