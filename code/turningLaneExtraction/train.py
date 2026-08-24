from datetime import datetime
import math
from decimal import Decimal
import json
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))


from PIL import Image  	# pyright: ignore[reportMissingImports]
import numpy as np		# pyright: ignore[reportMissingImports]
#from subprocess import Popen
#import tensorflow as tf

from framework.training import TrainingFramework
#from dataloader import Dataloader, ParallelDataLoader
from dataloader import ParallelDataLoader
from model import LinkModel


class Train(TrainingFramework):
	def __init__(self, mode="seg"):
		self.mode = mode
		self.image_size = 640
		self.batch_size = 4		# dky - original: 8
		self.datafolder = "../dataset_training"
		self.training_range = []
		dataset_split = json.load(open("../split_all.json"))
		
		for tid in dataset_split["training"]:
			for i in range(9):
				self.training_range.append("_%d" % (tid*9+i))
	
		self.instance = f"_turningLaneExtraction_{self.image_size}_resnet34_{mode}-{timestamp}"
		
		self.modelfolder = f"model{self.instance}"
		self.validationfolder = f"validation{self.instance}"

		os.makedirs(self.modelfolder, exist_ok=True)
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
		self.model = LinkModel(sess, self.image_size, batchsize=self.batch_size)
		
		return self.model

	def getBatch(self, dataloader):
		return dataloader.getBatch(self.batch_size)


	def train(self, batch, lr):
		self.counter += 1
		ret = self.model.train(batch[0], batch[1], batch[2], batch[3], batch[4], lr)
	
		return ret


	def preload(self, dataloader, step):
		if step > 0 and step % 50 == 0:
			dataloader.preload()


	# placeholder methods
	def getLoss(self, result):
		if math.isnan(result[0]):
			print(f"loss is nan ...")
			exit()

		return result[0]

	def getProgress(self, step):
		return step / float(self.epochsize)

	def saveModel(self, step, progress=None):
		save_every_epochs = 50
		save_every_steps = 1000

		cur_epoch = int(progress)
		cur_epoch_actual = round(progress, 2)

		end_epoch = int(self.epochsize)
		end_epoch_actual = round(self.epochsize, 2)

		save_epoch = (cur_epoch_actual * 100) % save_every_epochs		# dky: avoid floating point ops precision issues

		#print(f">>> progress={progress}, save_epoch={save_epoch} (cur_epoch_actual={cur_epoch_actual})")
		#print(f"step: {step} cur_epoch={cur_epoch}: save-epoch={cur_epoch % save_every_epochs}, save_every_epoch: {save_every_epochs} / cur_epoch={cur_epoch} / end_epoch={end_epoch}")
		#print(f"step {step} epoch:{cur_epoch_actual} ({cur_epoch}), checkpoint={saved_checkpoint} save-epoch={cur_epoch % save_every_epochs} / total epochs={end_epoch}")

		#if step > 0 and step % (self.epochsize * 5) == 0:
		if step > 500 and cur_epoch > 0 and (save_epoch == 0 or cur_epoch >= end_epoch):
			save_path = os.path.join(self.modelfolder, f"model{cur_epoch}")
			if not os.path.isfile(save_path):
				self.model.saveModel(save_path)

		return False	# do not stop training

	def visualization(self, step, result=None, batch=None):
		direction_img = np.zeros((self.image_size, self.image_size, 3))
		
		if step % 100 == 0:
			ind = ((step // 100) * self.batch_size) % 128
			for i in range(self.batch_size):
				Image.fromarray(((batch[0][i,:,:,:] + 0.5) * 255).astype(np.uint8)).save(self.validationfolder + "/input%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,0:3]) * 127 + 127).astype(np.uint8)).save(self.validationfolder + "/connector1%d.jpg" % (ind+i))
				Image.fromarray(((batch[1][i,:,:,3:6]) * 127 + 127).astype(np.uint8)).save(self.validationfolder + "/connector2%d.jpg" % (ind+i))
				
				Image.fromarray(((batch[2][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/target%d.jpg" % (ind+i))
				Image.fromarray(((result[1][i,:,:,0]) * 255).astype(np.uint8)).save(self.validationfolder + "/output%d.jpg" % (ind+i))

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
	now = datetime.now()
	timestamp = now.strftime("%Y%m%d-%H%M")

	#trainer = Train(sys.argv[1])
	trainer = Train()
	epochsisze = trainer.epochsize

	config = {}
	config["learningrate"] = 0.0001
	config["lr_decay"] = [0.1, 0.1]
	config["lr_decay_step"] = [epochsisze * 300, epochsisze * 350]
	config["step_init"] = 0
	config["step_max"] = epochsisze * 400 + 1
	config["use_validation"] = False
	#config["logfile"] = "log_%s.json" % trainer.instance
	config["logfile"] = f"log_{trainer.instance}.json"
	
	trainer.run(config)
