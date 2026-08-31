
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append("../cnnmodels")
import json
import math
#from decimal import Decimal

from PIL import Image			# pyright: ignore[reportMissingImports]
import numpy as np  			# pyright: ignore[reportMissingImports]
#from subprocess import Popen 	# dky: use standard os.makedirs instead

from framework.training import TrainingFramework
from dataloader import  ParallelDataLoader
from model import LaneModel



class Train(TrainingFramework):
	def __init__(self):
		self.image_size = 640
		self.batch_size = 4
		self.datafolder = "../dataset_training"
		self.training_range = []
		self.use_sdmap = False
		self.backbone = sys.argv[1] # resnet34v3

		dataset_split = json.load(open("../split_all.json"))

		for tid in dataset_split["training"]:
			for i in range(9):
				self.training_range.append("_%d" % (tid*9+i))

		self.instance = "_laneExtraction_run1_640_%s_500ep" % self.backbone

		if self.use_sdmap:
			self.instance += "_withsdmap"

		self.modelfolder = f"model{self.instance}"
		self.validationfolder = f"validation{self.instance}"

		#Popen("mkdir -p " + self.modelfolder, shell=True).wait()
		os.makedirs(self.modelfolder, exist_ok=True)

		#Popen("mkdir -p " + self.validationfolder, shell=True).wait()
		os.makedirs(self.validationfolder, exist_ok=True)

		self.counter = 0
		self.disloss = 0

		self.epochsize = len(self.training_range) * 2048 * 2048 / (self.batch_size * self.image_size * self.image_size)

		pass

	def createDataloader(self, mode):
		self.dataloader = ParallelDataLoader(self.datafolder, self.training_range, image_size=self.image_size)
		self.dataloader.preload()
		return self.dataloader

	def createModel(self, sess):
		self.model = LaneModel(sess, self.image_size, batchsize=self.batch_size, sdmap=self.use_sdmap, backbone=self.backbone)
		return self.model

	def getBatch(self, dataloader):
		return dataloader.getBatch(self.batch_size)

	def train(self, batch, lr):
		self.counter += 1
		ret = self.model.train(batch[0], batch[1], batch[2],batch[3], lr, sdmap=batch[-1])
		return ret


	def preload(self, dataloader, step):
		if step > 0 and step % 50 == 0:
			dataloader.preload()


	# placeholder methods
	def getLoss(self, result):
		if math.isnan(result[0]):
			print(f">> loss is nan ({result[0]})...")
			exit()

		return result[0]


	def getProgress(self, step):
		return step / float(self.epochsize)


	def saveModel(self, step, progress=None):
		save_every_epochs = 10
		save_every_steps = 1000

		cur_epoch = int(progress)
		cur_epoch_actual = round(progress, 2)

		end_epoch = int(self.epochsize)

		#is_epoch_end = (epoch > 0) and (step % self.epochsize == 0)
		#is_interval_epoch = (epoch > 0) and (epoch % save_every_epochs == 0)

		#if step % (self.epochsize * 10) == 0:	# dky: original from author
		if step > 0 and cur_epoch > 0 and (step % save_every_epochs == 0 or cur_epoch == end_epoch):
			print(f" >>> in saveModel(): step={step}, epochsize={self.epochsize}, step//epochsize={step // self.epochsize} (saving) ***")
			#self.model.saveModel(self.modelfolder + "/model%d" % (step // (self.epochsize)))	# dky: old school, not f-strings
			save_path = os.path.join(self.modelfolder, f"model{cur_epoch}")
			if not os.path.isfile(save_path):
				self.model.saveModel(save_path)

		return False		# do not stop training

	def visualization(self, step, result=None, batch=None):
		direction_img = np.zeros((self.image_size, self.image_size, 3))

		if step % 100 == 0:
			ind = ((step // 100) * self.batch_size) % 128
			#batch[3] = np.clip(batch[3], -1, 1)
			#result[1] = np.clip(result[1], -1, 1)

			for i in range(self.batch_size):
				Image.fromarray(((batch[0][i,:,:,:] + 0.5) * 255).astype(np.uint8)).save(self.validationfolder + "/input%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/mask%d.jpg" % (ind+i))
				Image.fromarray(((batch[2][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/target%d.jpg" % (ind+i))

				Image.fromarray(((batch[4][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/sdmap%d.jpg" % (ind+i))

				direction_img[:,:,2] = batch[3][i,:,:,0] * 127 + 127
				direction_img[:,:,1] = batch[3][i,:,:,1] * 127 + 127
				direction_img[:,:,0] = 127

				Image.fromarray(direction_img.astype(np.uint8)).save(self.validationfolder + "/targe_direction%d.jpg" % (ind+i))

				Image.fromarray(((result[1][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/output%d.jpg" % (ind+i))

				direction_img[:,:,2] = np.clip(result[1][i,:,:,1],-1,1) * 127 + 127
				direction_img[:,:,1] = np.clip(result[1][i,:,:,2],-1,1) * 127 + 127
				direction_img[:,:,0] = 127

				Image.fromarray(direction_img.astype(np.uint8)).save(self.validationfolder + "/output_direction%d.jpg" % (ind+i))

		return False


if __name__ == "__main__":
	trainer = Train()
	epochsisze = trainer.epochsize

	config = {}
	config["learningrate"] = 0.001
	config["lr_decay"] = [0.1, 0.1]
	config["lr_decay_step"] = [epochsisze * 350, epochsisze * 450]
	config["step_init"] = 0
	config["step_max"] = epochsisze * 500 + 1
	config["use_validation"] = False
	config["logfile"] = "log_lane_%s.json" % trainer.instance

	trainer.run(config)
