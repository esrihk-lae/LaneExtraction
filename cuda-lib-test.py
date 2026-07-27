# python2 and python3

import ctypes
import site
import os
import sys
import tensorflow as tf


#print(sys.version_info)
if sys.version_info < (3, 0):
    sys.exit("Error: This script requires Python 3.0 or higher.")


# Dynamically load the library from your local python packages
try:
    cusolver_path = f"{site.getsitepackages()[0]}/nvidia/cusolver/lib/libcusolver.so.11"
    ctypes.CDLL(cusolver_path, mode=ctypes.RTLD_GLOBAL)
    print(f"{cusolver_path}")
except IndexError:
    pass

print(f"\n")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"\n")


from tensorflow.python.client import device_lib
print("\n--- Available Devices ---")
print(device_lib.list_local_devices())


print(f"Is GPU Available? {tf.test.is_gpu_available()}")
#print(tf.config.list_physical_devices('GPU'))

