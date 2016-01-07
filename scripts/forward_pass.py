#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import IPython
import sys
import pickle
import caffe
import argparse
import cv2

import utils
import constants

sys.path.insert(0, constants.CAFFE_ROOT + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class CNNFeatureExtractor:
	def __init__(self, net_name = "AlexNet"):
		self.net = None
		self.transformer = None
		self.init_caffe(net_name)

	def forward_pass(self, PATH_TO_DATA, annotations, list_of_layers = constants.alex_net_layers,
		sampling_rate = 1, batch_size = -1, LCD = False, no_plot_mode = False):
		return self.process_individual_frames(PATH_TO_DATA, annotations,
			list_of_layers, sampling_rate)

	def process_individual_frames(self, PATH_TO_DATA, annotations, list_of_layers, sampling_rate):
		X = {}
		map_index_data = pickle.load(open(annotations, "rb"))

		segments = utils.get_chronological_sequences(map_index_data)
		for seg in segments:
			print str(seg)
			frm_num = seg[0]
			while frm_num <= seg[1]:
				print frm_num, annotations
				im = caffe.io.load_image(utils.get_full_image_path(PATH_TO_DATA, frm_num))
				self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
				out = self.net.forward()
				for layer in list_of_layers:
					if layer == 'input':
						data = cv2.imread(full_image_path)
					else:
						data = self.net.blobs[layer].data[0]
					data = utils.flatten(data)
					utils.dict_insert(layer, data, X)
				frm_num += sampling_rate
		return X


	def init_caffe(self, net_name):
		caffe.set_mode_gpu()
		net_params = constants.NET_PARAMS[net_name]
		self.net = caffe.Net(net_params[0], net_params[1], caffe.TEST)
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_transpose('data', (2,0,1))
		# self.transformer.set_mean('data', np.load(constants.CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
		self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
		if net_name == "AlexNet":
			self.net.blobs['data'].reshape(1,3,227,227)
		else:
			self.net.blobs['data'].reshape(1,3,224,224)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("net", help = "Net, Choose from: AlexNet, VGG, VGG_SOS")
	parser.add_argument("figure_name", help = "Please specify MAIN file name")
	parser.add_argument("PATH_TO_DATA", help="Please specify the path to the images")
	parser.add_argument("annotations", help = "Annotated frames")
	parser.add_argument("--PATH_TO_DATA_2", help="Please specify the path to second set of images", default = None)
	parser.add_argument("--annotations_2", help = "Annotated frames for second set of images", default = None)
	args = parser.parse_args()
	fe = CNNFeatureExtractor(args.net)
	layers = constants.NET_PARAMS[args.net][2]

	X = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = layers, sampling_rate = 1)
	utils.plot_all_layers(X, args.net, args.figure_name, list_of_layers = layers)

