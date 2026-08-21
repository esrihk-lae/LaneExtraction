import sys

from PIL import Image
import imageio.v2 as imageio
import numpy as np
from subprocess import Popen
import tensorflow as tf
#import scipy.ndimage

from model import LaneModel


inputfile = sys.argv[1]
outputfolder = sys.argv[2]
backbone = sys.argv[3]
windowsize1 = 256
windowsize2 = 512									# dky: test - per error: could not broadcast input array from shape (640,512) into shape (640,640)
#windowsize2 = 640
#cnninput = 512
cnninput = 640		#orig


margin = (cnninput - windowsize1) // 2
margin2 = (cnninput - windowsize2) // 2

Popen("mkdir -p " + outputfolder, shell=True).wait()

#img = scipy.ndimage.imread(inputfile)				# dky: orig
img = imageio.imread(inputfile, mode='RGB')			# dky: change to imageio, read as RGB?

#sdmap = scipy.ndimage.imread(inputfile.replace("sat", "sdmap"))
sdmap = imageio.imread(inputfile.replace("sat", "sdmap"), mode='L')		# dky: change to imageio; sdmap test read as mode L
# dky TODO: need to figure out what sdmap is to align read mode -- what color mode is this?

#img = (img.astype(np.float) / 255.0 - 0.5) * 0.81 		# dky: test update for np.float -> float
img = (img.astype(float) / 255.0 - 0.5) * 0.81

#sdmap = sdmap.astype(np.float) / 255.0		# dky: test update for np.float -> float
sdmap = sdmap.astype(float) / 255.0
dim = np.shape(img)

# np.pad(array, pad_width, mode='constant', **kwargs)
#print(f">>> img={img}, \nmargin={margin}, margin2={margin2}, windowsize1={windowsize1}, windowsize2={windowsize2}, cnninput={cnninput}")
img = np.pad(img, ((margin, margin), (margin, margin), (0,0)), 'constant')
sdmap = np.pad(sdmap, ((margin, margin), (margin, margin)), 'constant')				# dky ????

mask = np.zeros((cnninput,cnninput,3))
for i in range((windowsize2 - windowsize1) // 2 ):
	r = i / float((windowsize2 - windowsize1) // 2)
	mask[margin2+i:-(margin2+i-1),margin2+i:-(margin2+i-1),:] = r


output = np.zeros_like(img)
weights = np.zeros_like(img) + 0.0001

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
	model = LaneModel(sess, cnninput, batchsize=1, sdmap=False, backbone=backbone)
	#model.restoreModel("modelrun3_640_resnet34v3/model196900")
	#model.restoreModel("modelfinetune_run1_640_resnet34v3/model52215")
	#model.restoreModel("modelfinetune_run2_640_resnet34v3/model52215")
	#model.restoreModel("./model_laneExtraction_run1_640_resnet34v3_500ep/model52215")

	x_in = np.zeros((1, cnninput, cnninput, 3))
	x_in2 = np.zeros((1, cnninput, cnninput, 1))


	for model_ep in [499]:
		if backbone == "resnet34v3":
			#model.restoreModel("model_4cities_run2_640_%s_500ep/model%d" % (backbone, model_ep))
			#model_laneExtraction_run1_640_resnet34v3_500ep
			model.restoreModel("./model_laneExtraction_run1_640_%s_500ep/model%d.0" % (backbone, model_ep))
			#model.restoreModel("./model_laneExtraction_run1_640_%s_500ep/checkpoint" % (backbone))
		else:
			#model.restoreModel("model_4cities_run1_640_%s_500ep/model%d" % (backbone, model_ep))
			model.restoreModel("./model_laneExtraction_run1_640_%s_500ep/model%d.0" % (backbone, model_ep))

		#print(f">>> type {type(sdmap)}")		# dky: debug

		# sliding window inference
		for i in range(dim[0] // windowsize1):
			for j in range(dim[1] // windowsize1):
				#print(f">>> i={i}, j={j}")
				r = i * windowsize1
				c = j * windowsize1

				#print(f">>> row r={r}, col c={c}")
				x_in[0,:,:,:] = img[r:r+cnninput, c:c+cnninput,:]
				#x_in2[0,:,:,0] = sdmap[r:r+cnninput, c:c+cnninput] 	# dky ??? lol what???

				x_out = model.infer(x_in)[0]

				output[r:r+cnninput, c:c+cnninput,:] += x_out[0,:,:,:] * mask
				weights[r:r+cnninput, c:c+cnninput,:] += mask[:,:,0:1]

	output = np.divide(output, weights)

	output = output[margin:-margin, margin:-margin,:]
	Image.fromarray(((output[:,:,0]) * 255).astype(np.uint8)).save(outputfolder + "/seg.png")

	direction_img = np.zeros(dim, dtype=np.uint8)

	direction_img[:,:,2] = np.clip(output[:,:,1],-1,1) * 127 + 127
	direction_img[:,:,1] = np.clip(output[:,:,2],-1,1) * 127 + 127
	direction_img[:,:,0] = 127

	Image.fromarray(direction_img.astype(np.uint8)).save(outputfolder + "/direction.png")



