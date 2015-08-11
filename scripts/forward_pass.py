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
import encoding
import constants
import lcd

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
		if no_plot_mode:
			return self.process_individual_frames_2(PATH_TO_DATA, annotations,
				list_of_layers, sampling_rate)
		if batch_size == -1:
			return self.process_individual_frames(PATH_TO_DATA, annotations,
				list_of_layers, sampling_rate, LCD)
		return self.process_batch(PATH_TO_DATA, annotations, list_of_layers,
			sampling_rate, batch_size, LCD)

	def process_batch(self, PATH_TO_DATA, annotations, list_of_layers, sampling_rate, batch_size, LCD):
		i = 0
		label_map = {}
		frm_map = {}
		X = {}
		map_index_data = pickle.load(open(annotations, "rb"))

		for index in map_index_data:
			segments = map_index_data[index]
			# print "Processing images for label " + str(index)
			for seg in segments:
				# print str(seg)
				frm_num = seg[0]
				b = 1 #Running count of num frames in batch
				batch_data = {}
				while frm_num <= (seg[1] + batch_size):
					# Initialize Batch
					if b == 1:
						label_map[i] = index
						frm_map[i] = frm_num
					# Process frames and build up features in batches
					im = caffe.io.load_image(utils.get_full_image_path(PATH_TO_DATA, frm_num))
					self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
					out = self.net.forward()
					for layer in list_of_layers:
						if LCD:
							if layer == 'input':
								print "ERROR: Cannot do LCD on input layer"
								sys.exit()
							data = self.net.blobs[layer].data[0]
							data = lcd.LCD(data)
							utils.dict_insert(layer, data, batch_data, axis = 0)
						else:
							if layer == 'input':
								data = cv2.imread(full_image_path)
							else:
								data = self.net.blobs[layer].data[0]
							data = utils.flatten(data)
							utils.dict_insert(layer, data, batch_data, axis = 1)
					if b == batch_size:
						print("Batch %3d" % i)
						b = 0
						i += 1
						# Concatenate with main data dictionary
						for layer in list_of_layers:
							data = encoding.encode_VLAD(batch_data[layer] , 5)
							utils.dict_insert(layer, data, X)
						batch_data = {}

					b += 1
					frm_num += sampling_rate
		return X, label_map, frm_map

	def process_individual_frames(self, PATH_TO_DATA, annotations, list_of_layers, sampling_rate, LCD):
		i = 0
		label_map = {}
		frm_map = {}
		X = {}
		map_index_data = pickle.load(open(annotations, "rb"))

		IPython.embed()

		for index in map_index_data:
			segments = map_index_data[index]
			print "Processing images for label " + str(index)
			for seg in segments:
				print str(seg)
				frm_num = seg[0]
				while frm_num <= seg[1]:
					print frm_num
					frm_map[i] = frm_num
					label_map[i] = index
					im = caffe.io.load_image(utils.get_full_image_path(PATH_TO_DATA, frm_num))
					self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
					out = self.net.forward()
					IPython.embed()
					for layer in list_of_layers:
						if layer == 'input':
							data = cv2.imread(full_image_path)
						else:
							data = self.net.blobs[layer].data[0]
						data = utils.flatten(data)
						utils.dict_insert(layer, data, X)
					frm_num += sampling_rate
					i += 1
		return X, label_map, frm_map

	def process_individual_frames_2(self, PATH_TO_DATA, annotations, list_of_layers, sampling_rate):
		X = {}
		map_index_data = pickle.load(open(annotations, "rb"))

		segments = utils.get_chronological_sequences(map_index_data)
		for seg in segments:
			print str(seg)
			frm_num = seg[0]
			while frm_num <= seg[1]:
				print frm_num
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
	parser.add_argument("--hypercolumns", help = "Hypercolumns for conv 3 and conv 4", default = False)
	parser.add_argument("--vlad", help = "VLAD experiments", default = False)
	parser.add_argument("--batch_size", help = "Batch size for temporal batches", default = -1)
	parser.add_argument("--LCD", help = "Batch size for temporal batches")
	args = parser.parse_args()
	fe = CNNFeatureExtractor(args.net)
	layers = constants.NET_PARAMS[args.net][2]
	if args.PATH_TO_DATA_2 or args.annotations_2:
		if args.PATH_TO_DATA_2 and args.annotations_2:

			# layers = constants.vgg_sos_net_layers[11:]
			# batch_size = int(args.batch_size) if args.batch_size else -1
			# encoding_func = encoding.encode_cluster_normalize if args.vlad else None
			print "--------- Forward Pass #1 ---------"
			X1, label_map_1, frm_map_1 = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = layers)
			print "--------- Forward Pass #2 ---------"
			X2, label_map_2, frm_map_2 = fe.forward_pass(args.PATH_TO_DATA_2, args.annotations_2, list_of_layers = layers)

			utils.plot_all_layers_joint(X1, args.net, label_map_1, frm_map_1, X2, label_map_2, frm_map_2, args.figure_name, layers = layers)
		else:
			print "ERROR: Please provide both annotations and the path for second set of images"
			sys.exit()
	elif args.hypercolumns:
		print "Processing Hypercolumns"
		hypercolumns_layers = ['conv3', 'conv4']
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, hypercolumns_layers)
		X_hc = utils.make_hypercolumns_vector(hypercolumns_layers, X)
		utils.plot_hypercolumns(X_hc, args.net, label_map, frm_map, args.figure_name, hypercolumns_layers)

		hypercolumns_layers = ['conv2','conv3', 'conv4']
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, hypercolumns_layers)
		X_hc = utils.make_hypercolumns_vector(hypercolumns_layers, X)
		utils.plot_hypercolumns(X_hc, args.net, label_map, frm_map, args.figure_name, hypercolumns_layers)
	elif args.vlad:
		layers = ['conv4', 'conv3']
		k_values = [1, 5, 10]
		pc_values = [5, 100, 200]
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = layers, sampling_rate = 1)
		utils.vlad_experiment(X, k_values, pc_values, label_map, frm_map, args.figure_name, list_of_layers = layers)

	elif args.batch_size != -1 and not args.LCD:
		layers = ['conv3', 'conv4', 'conv5', 'pool5']
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations,list_of_layers = layers, sampling_rate = 1, batch_size = int(args.batch_size))
		utils.plot_all_layers(X, args.net, label_map, frm_map, args.figure_name, list_of_layers = layers)

	elif args.LCD:
		if args.batch_size == -1:
			print "ERROR: Please provide both batch size and LCD"
			sys.exit()
		layers = ['conv5_1', 'conv5_2', 'conv5_3' ]
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = layers, sampling_rate = 1, batch_size = int(args.batch_size), LCD = True)
		utils.plot_all_layers(X, args.net, label_map, frm_map, args.figure_name, list_of_layers = layers)

	else:
		print "Plotting Individual frames"
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = layers)
		utils.plot_all_layers(X, args.net, label_map, frm_map, args.figure_name, list_of_layers = layers)

