#!/bin/bash
# dky
# 2026-07
#

# note: not intended to be run directly; 
# source me instead.

unset LD_LIBRARY_PATH

#export LD_LIBRARY_PATH=$(find /home/dky/src/LaneExtraction -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)
#export LD_LIBRARY_PATH=$(find /usr/local/cuda -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)
#export LD_LIBRARY_PATH=$(find /usr/local -name "*.so*" | xargs dirname | sort -u | paste -d ":" -s -)

echo $LD_LIBRARY_PATH


python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

