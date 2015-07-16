import numpy as np
import matplotlib.pyplot as plt
import IPython
import sys
import pickle
import caffe
import argparse
import utils
import cv2
import utils

sys.path.insert(0, utils.CAFFE_ROOT + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class FeatureExtractor:
	def __init__(self):
		self.net = None
		self.transformer = None
		self.init_caffe()

	def forward_pass(self, PATH_TO_DATA, annotations):
		i = 0
		label_map = {}
		index_map = {}
		X = {}
		map_index_data = pickle.load(open(annotations, "rb"))

		self.net.blobs['data'].reshape(1,3,227,227)

		for index in map_index_data:
			segments = map_index_data[index]
			print "Processing images for label " + str(index)
			for seg in segments:
				print str(seg)
				frm_num = seg[0]
				while frm_num <= seg[1]:
					print frm_num
					index_map[i] = frm_num
					label_map[i] = index
					image_jpeg = PATH_TO_DATA + utils.get_frame_fig_name(frm_num)
					self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(image_jpeg))
					out = self.net.forward()
					plt.imshow(self.transformer.deprocess('data', self.net.blobs['data'].data[0]))
					for layer in utils.alex_net_layers:
						if layer == 'input':
							data = cv2.imread(image_jpeg)
						else:
							data = self.net.blobs[layer].data[0]
						data = data.flatten()
						data = data.reshape(1, data.shape[0])
						if layer not in X:
							X[layer] = data
						else:
							X_layer = X[layer]
							X_layer = np.concatenate((X_layer, data), axis = 0)
							X[layer] = X_layer
					frm_num += 2
					i += 1
		return X, label_map, index_map

	def init_caffe(self):
		caffe.set_mode_gpu()
		self.net = caffe.Net(utils.CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt',
		                utils.CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
		                caffe.TEST)
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_transpose('data', (2,0,1))
		self.transformer.set_mean('data', np.load(utils.CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
		self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("figure_name", help = "Please specify MAIN file name")
	parser.add_argument("PATH_TO_DATA", help="Please specify the path to the images")
	parser.add_argument("annotations", help = "Annotated frames")
	parser.add_argument("--PATH_TO_DATA_2", help="Please specify the path to second set of images", default = None)
	parser.add_argument("--annotations_2", help = "Annotated frames for second set of images", default = None)
	args = parser.parse_args()
	fe = FeatureExtractor()
	if args.PATH_TO_DATA_2 or args.annotations_2:
		if args.PATH_TO_DATA_2 and args.annotations_2:
			print "--------- Forward Pass #1 ---------"
			X1, label_map_1, index_map_1 = fe.forward_pass(args.PATH_TO_DATA, args.annotations)
			print "--------- Forward Pass #2 ---------"
			X2, label_map_2, index_map_2 = fe.forward_pass(args.PATH_TO_DATA_2, args.annotations_2)
			utils.plot_all_layers_joint(X1, label_map_1, index_map_1, X2, label_map_2, index_map_2, args.figure_name)
		else:
			print "ERROR: Please provide both annotations and the path for second set of images"
			sys.exit()
	else:
		X, label_map, index_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations)
		utils.plot_all_layers(X, label_map, index_map, args.figure_name)

