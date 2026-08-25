from datetime import datetime
import math
from decimal import Decimal
import json
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))


from PIL import Image				# pyright: ignore[reportMissingImports]
import numpy as np					# pyright: ignore[reportMissingImports]
#from subprocess import Popen
#import tensorflow as tf			# dky: unused import

from framework.training import TrainingFramework
#from dataloader import Dataloader, ParallelDataLoader		# dky: original
from dataloader import ParallelDataLoader
from model import LinkModel



class Train(TrainingFramework):
	def __init__(self, mode="seg"):
		self.mode = mode
		self.image_size = 640
		self.batch_size = 1						# dky - original: 8, unofficial sets this to 1
		self.datafolder = "../dataset_training"
		self.training_range = []
		dataset_split = json.load(open("../split_all.json"))

		for tid in dataset_split["training"]:
			for i in range(9):
				self.training_range.append("_%d" % (tid*9+i))

		#self.instance = "link_run6_640_resnet34v3" # gt direction
		#self.instance = "_turningLaneValidation_run1_640_resnet34_500ep" + self.mode
		self.instance = f"_turningLaneValidation_{self.image_size}_resnet34_500ep_{self.mode}-{timestamp}"

		self.modelfolder = f"model{self.instance}"
		self.validationfolder = f"validation{self.instance}"

		os.makedirs(self.modelfolder, exist_ok=True)
		os.makedirs(self.validationfolder, exist_ok=True)

		self.counter = 0
		self.disloss = 0

		self.epochsize = len(self.training_range) * 2048 * 2048 / (self.batch_size * self.image_size * self.image_size)
		#print(f">>> init(): self.epochsize={self.epochsize}")
		pass

	def createDataloader(self, mode):
		self.dataloader = ParallelDataLoader(self.datafolder, self.training_range, image_size=self.image_size)
		self.dataloader.preload()
		return self.dataloader

	def createModel(self, sess):
		self.model = LinkModel(sess, self.image_size, batchsize=self.batch_size)

		return self.model

	def getBatch(self, dataloader):
		return dataloader.getBatch(self.batch_size)

	def train(self, batch, lr):
		self.counter += 1
		ret = self.model.train(batch[0], batch[1], batch[2], batch[3], batch[4], lr)

		return ret

	## dky: preload()
	def preload(self, dataloader, step):
		if step > 0 and step % 50 == 0:
			dataloader.preload()


	# placeholder methods
	def getLoss(self, result):
		if math.isnan(result[0]):
			print(f"ERROR: loss is nan ...")
			exit()

		self.logvalue("segloss", result[1])
		self.logvalue("classloss", result[2])

		return result[0]

	def getProgress(self, step):
		return step / float(self.epochsize)
	

	def saveModel(self, step, progress=None):
		save_every_epochs = 10
		save_every_steps = 1000

		cur_epoch = int(progress)
		cur_epochsize = int(self.epochsize)

		end_epoch = int(self.epochsize)

		#if step > 0 and step % (self.epochsize * 5) == 0:
		#if step > 0 and (step % 1000 == 0 or cur_epoch == cur_epochsize):
		if step > 0 and cur_epoch > 0 and (step % save_every_epochs == 0 or cur_epoch == end_epoch):
			#self.model.saveModel(self.modelfolder + "/model%d" % step)
			save_path = os.path.join(self.modelfolder, f"model{cur_epoch}")
			if not os.path.isfile(save_path):
				self.model.saveModel(save_path)

		return False


	def visualization(self, step, result=None, batch=None):
		direction_img = np.zeros((self.image_size, self.image_size, 3))

		if step % 100 == 0:
			ind = ((step // 100) * self.batch_size) % 128
			for i in range(self.batch_size):
				Image.fromarray(((batch[0][i,:,:,:] + 0.5) * 255).astype(np.uint8)).save(self.validationfolder + "/input%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,0:3]) * 127 + 127).astype(np.uint8)).save(self.validationfolder + "/connector1%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,3:6]) * 127 + 127).astype(np.uint8)).save(self.validationfolder + "/connector2%d.jpg" % (ind+i))

				Image.fromarray(((batch[2][i,:,:,1]) * 255).astype(np.uint8)).save(self.validationfolder + "/target1%d.jpg" % (ind+i))
				Image.fromarray(((result[3][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/output1%d.jpg" % (ind+i))
				Image.fromarray(((batch[2][i,:,:,2]) * 255).astype(np.uint8)).save(self.validationfolder + "/target2%d.jpg" % (ind+i))
				Image.fromarray(((result[3][i,:,:,1]) * 255).astype(np.uint8)).save(self.validationfolder + "/output2%d.jpg" % (ind+i))

				with open(self.validationfolder + "/label%d.txt" % (ind+i), "w") as fout:
					fout.write("%f %f \n" % (batch[3][i,0], result[4][i,0]))

				def norm(x):
					#return x
					amin = np.amin(x)
					amin = 0
					amax = np.amax(x)

					x = (x - amin) / max(0.00001, (amax - amin))
					return x


				direction_img[:,:,2] = np.clip(batch[4][i,:,:,0],-1,1) * 127 + 127
				direction_img[:,:,1] = np.clip(batch[4][i,:,:,1],-1,1) * 127 + 127
				direction_img[:,:,0] = 127

				direction_img[:,:,0] += batch[1][i,:,:,0] * 255 + 127
				direction_img[:,:,1] += batch[1][i,:,:,3] * 255 + 127
				direction_img[:,:,2] += batch[1][i,:,:,6] * 255 + 127

				direction_img = np.clip(direction_img, 0, 255)

				Image.fromarray(direction_img.astype(np.uint8)).save(self.validationfolder + "/direction%d.jpg" % (ind+i))

		return False


if __name__ == "__main__":

	#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify the GPU device to use (e.g., GPU 0)
	#os.environ['TF_ALLOW_IOLIBS'] = '0' # Disable TensorFlow I/O libraries to avoid OOM issues
	#os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'  # Disable cuDNN autotune to avoid OOM issues

	#from tensorflow.python.client import device_lib

	# tf GPU configuration - see infer.py
	# Log exactly which device (CPU or GPU) each operation is assigned to
	#config = tf.ConfigProto(log_device_placement=True)

	# Initialize RunOptions and turn on the OOM allocation report
	# run_options = tf.compat.v1.RunOptions()
	# run_options.report_tensor_allocations_upon_oom = True

	# Initialize RunMetadata to collect the execution metrics
	# run_metadata = tf.compat.v1.RunMetadata()

	# Memory Management
	#config.gpu_options.allow_growth = False
	#config.gpu_options.per_process_gpu_memory_fraction = 0.5

	# Print the name of the primary GPU device if available
	#print(f"Is GPU available:  {tf.test.is_gpu_available()}")
	#print(f"Primary GPU Device Name: {tf.test.gpu_device_name()}")

	#gpus = tf.config.list_physical_devices('GPU')
	# Replace list_physical_devices with the legacy v1 equivalent:
	#gpus = tf.config.experimental.list_devices()
	#gpus = device_lib.list_local_devices()
	#print(device_lib.list_local_devices())
	#print(tf.config.experimental.list_physical_devices('GPU'))

	now = datetime.now()
	timestamp = now.strftime("%Y%m%d-%H%M")

	#trainer = Train(sys.argv[1])
	trainer = Train()
	epochsisze = trainer.epochsize

	config = {}
	config["learningrate"] = 0.0001
	config["lr_decay"] = [0.1, 0.1]
	config["lr_decay_step"] = [epochsisze * 350, epochsisze * 450]
	config["step_init"] = 0
	config["step_max"] = epochsisze * 500 + 1
	config["use_validation"] = False
	config["logfile"] = "log_%s.json" % trainer.instance

	try:
		trainer.run(config)
	finally:
		if hasattr(trainer, 'dataloader') and hasattr(trainer.dataloader, 'stop'):
			trainer.dataloader.stop()
