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
import encoding

sys.path.insert(0, utils.CAFFE_ROOT + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class FeatureExtractor:
	def __init__(self):
		self.net = None
		self.transformer = None
		self.init_caffe()

	def forward_pass(self, PATH_TO_DATA, annotations, list_of_layers = utils.alex_net_layers, sampling_rate = 1, batch_size = -1):
		if batch_size == -1:
			return self.process_individual_frames(PATH_TO_DATA, annotations, list_of_layers, sampling_rate) 
		return self.process_batch(PATH_TO_DATA, annotations, list_of_layers, sampling_rate, batch_size)

	def process_batch(self, PATH_TO_DATA, annotations, list_of_layers, sampling_rate, batch_size):
		i = 0
		label_map = {}
		frm_map = {}
		X = {}
		map_index_data = pickle.load(open(annotations, "rb"))

		self.net.blobs['data'].reshape(1,3,227,227)

		for index in map_index_data:
			segments = map_index_data[index]
			print "Processing images for label " + str(index)
			for seg in segments:
				print str(seg)
				frm_num = seg[0]
				b = 1 #Running count of num frames in batch
				batch_data = {}
				while frm_num <= (seg[1] + batch_size):
					if b == 1:
						label_map[i] = index
						frm_map[i] = frm_num

					# Process frames and build up features in batches
					self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(get_full_image_path(PATH_TO_DATA, frm_num)))
					out = self.net.forward()
					for layer in list_of_layers:
						if layer == 'input':
							data = cv2.imread(full_image_path)
						else:
							data = self.net.blobs[layer].data[0]
						data = data.flatten()
						data = data.reshape(1, data.shape[0])
						if layer not in batch_data:
							batch_data[layer] = data
						else:
							X_layer = batch_data[layer]
							X_layer = np.concatenate((X_layer, data), axis = 1)
							batch_data[layer] = X_layer

					if b == batch_size:
						b = 0
						i += 1

						# Time to concatenate with main data dictionary
						for layer in list_of_layers:
							if layer not in X:
								X[layer] = batch_data[layer]
							else:
								X_layer = X[layer]
								X_layer = np.concatenate((X_layer, batch_data[layer]), axis = 0)
								X[layer] = X_layer
						batch_data = {}

					# IPython.embed()
					b += 1
					frm_num += sampling_rate
		return X, label_map, frm_map

	def process_individual_frames(self, PATH_TO_DATA, annotations, list_of_layers, sampling_rate):
		i = 0
		label_map = {}
		frm_map = {}
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
					frm_map[i] = frm_num
					label_map[i] = index
					self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(get_full_image_path(PATH_TO_DATA, frm_num)))
					out = self.net.forward()
					for layer in list_of_layers:
						if layer == 'input':
							data = cv2.imread(full_image_path)
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
					frm_num += sampling_rate
					i += 1
		return X, label_map, frm_map

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
	parser.add_argument("--hypercolumns", help = "Hypercolumns for conv 3 and conv 4", default = False)
	parser.add_argument("--vlad", help = "VLAD experiments", default = False)
	parser.add_argument("--batch_size", help = "Batch size for temporal batches", default = -1)
	args = parser.parse_args()
	fe = FeatureExtractor()
	if args.PATH_TO_DATA_2 or args.annotations_2:
		if args.PATH_TO_DATA_2 and args.annotations_2:

			list_of_layers = ['conv2','conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7']
			batch_size = int(args.batch_size) if args.batch_size else -1
			encoding_func = encode_cluster_normalize if args.vlad else None
			print "--------- Forward Pass #1 ---------"
			X1, label_map_1, frm_map_1 = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = list_of_layers, sampling_rate = 1, batch_size = batch_size)
			print "--------- Forward Pass #2 ---------"
			X2, label_map_2, frm_map_2 = fe.forward_pass(args.PATH_TO_DATA_2, args.annotations_2, list_of_layers = list_of_layers, sampling_rate = 1, batch_size = batch_size)

			utils.plot_all_layers_joint(X1, label_map_1, frm_map_1, X2, label_map_2, frm_map_2, args.figure_name, encoding_func = encoding_func, layers = list_of_layers)
		else:
			print "ERROR: Please provide both annotations and the path for second set of images"
			sys.exit()
	elif args.hypercolumns:
		print "Processing Hypercolumns"
		hypercolumns_layers = ['conv3', 'conv4']
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, hypercolumns_layers)
		X_hc = utils.make_hypercolumns_vector(hypercolumns_layers, X)
		utils.plot_hypercolumns(X_hc, label_map, frm_map, args.figure_name, hypercolumns_layers)

		hypercolumns_layers = ['conv2','conv3', 'conv4']
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, hypercolumns_layers)
		X_hc = utils.make_hypercolumns_vector(hypercolumns_layers, X)
		utils.plot_hypercolumns(X_hc, label_map, frm_map, args.figure_name, hypercolumns_layers)
	elif args.vlad:
		layers = ['conv4', 'conv3']
		k_values = [1, 5, 10]
		pc_values = [5, 100, 200]
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations, list_of_layers = layers, sampling_rate = 1)
		utils.vlad_experiment(X, k_values, pc_values, label_map, frm_map, args.figure_name, list_of_layers = layers)

	elif args.batch_size:
		layers = ['conv3', 'conv4', 'conv5', 'pool5']
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations,list_of_layers = layers, sampling_rate = 1, batch_size = int(args.batch_size))
		IPython.embed()
		utils.plot_all_layers(X, label_map, frm_map, args.figure_name, list_of_layers = layers)

	else:
		X, label_map, frm_map = fe.forward_pass(args.PATH_TO_DATA, args.annotations)
		utils.plot_all_layers(X, label_map, frm_map, args.figure_name)

