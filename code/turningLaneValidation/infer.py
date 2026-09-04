import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

import tensorflow as tf

from model import LinkModel



class InferEngine():
    def __init__(self, modelpath="../../models/TODO", batchsize = 4):   # dky - original: batchsize = 8
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.80, log_device_placement=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.model = LinkModel(self.sess, 640, batchsize=batchsize)
        self.model.restoreModel(modelpath)

    def infer(self, sat=None, connector=None, direction = None):
        return self.model.infer(sat, connector, direction)


