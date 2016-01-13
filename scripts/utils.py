#!/usr/bin/env python

import IPython
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (metrics, manifold, datasets, decomposition, ensemble, lda,
	random_projection, preprocessing, covariance, cluster, neighbors)
import random
import os
import yaml
import pickle
import time
from decimal import Decimal

import encoding
import constants

current_milli_time = lambda: int(round(time.time() * 1000))

def cca(W, Z):
	"""
	Canonical Correlation Analysis (CCA): Returns rows of Z which are maximally correlated to W.
	"""
	print "CCA....."
	import matlab
	import matlab.engine as mateng

	eng = mateng.start_matlab()	

	W_mat = matlab.double(W.tolist())
	Z_mat = matlab.double(Z.tolist())

	[A, B, r, U, V, stats] = eng.canoncorr(W_mat, Z_mat, nargout = 6)

	Z = np.array(V)

	eng.quit()

	return Z

def grp(X, C = 100):
	"""
	Gaussian Random Projection (GRP): Projection of X into C dimensions. 
	"""
	print "GRP..."
	print X.shape
	print("Computing GaussianRandomProjection, using %3d components" % C)
	transformer = random_projection.GaussianRandomProjection(n_components = C)
	X_grp = transformer.fit_transform(X)
	print X_grp.shape
	return X_grp

def pca(X, PC = 2):
	"""
	Principal Components Analysis (PCA): Scaling followed by projection of X
    onto PC principal components
    """
	print "PCA....."
	print("Computing PCA embedding, using %3d principal components" % PC)
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	X_pca = decomposition.TruncatedSVD(n_components = PC).fit_transform(X_centered)
	return X_pca

def pca_incremental(X, PC = 2):
	print "PCA....."
	print("Incremental PCA, using %3d principal components" % PC)
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	X_pca = decomposition.IncrementalPCA(n_components=PC).fit_transform(X_centered)
	return X_pca

def tsne_pca(X):
	print("Computing PCA -> t-SNE embedding")
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	X_pca = decomposition.TruncatedSVD(n_components=100).fit_transform(X_centered)
	tsne = manifold.TSNE(init = 'pca')
	X_tsne = tsne.fit_transform(X_pca)
	return X_tsne

def tsne(X):
	print("Computing t-SNE embedding")
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	X_tsne = tsne.fit_transform(X_centered)
	return X_tsne

def make_hypercolumns_vector(hypercolumns_layers, X):
	X_hc = None
	for layer in hypercolumns_layers:
		if X_hc is None:
			X_hc = X[layer]
		else:
			X_hc = np.concatenate((X_hc, X[layer]), axis = 1)
	return X_hc

def plot_annotated_joint(X_joint, num_X1_pts, label_map_1, frm_map_1, label_map_2, frm_map_2, figure_name, title = None):
	x_min, x_max = np.min(X_joint, 0), np.max(X_joint, 0)
	X_joint = (X_joint - x_min) / (x_max - x_min)
	plt.figure()
	for i in range(X_joint.shape[0]):
		if i < num_X1_pts:
			frm_num = frm_map_1[i]
			plt.text(X_joint[i, 0], X_joint[i, 1], 'x', color = constants.color_map[label_map_1[i]],
				fontdict = {'weight': 'bold', 'size': 10})
		else:
			frm_num = frm_map_2[i - num_X1_pts]
			plt.text(X_joint[i, 0], X_joint[i, 1], 'o', color = constants.color_map[label_map_2[i - num_X1_pts]],
				fontdict = {'weight': 'bold', 'size': 10})
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)
	plt.savefig(constants.PATH_TO_SAVE_FIG + figure_name + '.jpg')

# Scale and visualize the embedding vectors
def plot_annotated_embedding(X, label_map, frm_map, figure_name, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	for i in range(X.shape[0]):
		frm_num = frm_map[i]
		plt.text(X[i, 0], X[i, 1], "x", color=constants.color_map[label_map[i]],
			fontdict = {'weight': 'bold', 'size': 10})
	plt.xticks([]), plt.yticks([])
 	if title is not None:
		plt.title(title)
	plt.savefig(constants.PATH_TO_SAVE_FIG + figure_name + '.jpg')

# Scale and visualize the embedding vectors
def plot_pylab_scatter(X, label_map, frm_map, figure_name, title=None):
	"""
	Plotting with pylab scatter.
	"""
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	points = {}
	for i in range(X.shape[0]):
		x_coord = X[i, 0]
		y_coord = X[i, 1]
		color = constants.color_map[label_map[i]]
		if color in points:
			(coords_x, coords_y) = points[color]
			coords_x.append(x_coord)
			coords_y.append(y_coord)
			points[color] = (coords_x, coords_y)
		else:
			coords_x = [x_coord,]
			coords_y = [y_coord,]
			points[color] = (coords_x, coords_y)
	for color in points:
		(coords_x, coords_y) = points[color]
		pl.scatter(np.array(coords_x), np.array(coords_y), color = color)
	pl.savefig(constants.PATH_TO_SAVE_FIG + figure_name + '.jpg')

# Scale and visualize the embedding vectors
def plot_scatter_continous(X, figure_name, title = None, colormap_name = "Accent"):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	coords_x = []
	coords_y = []
	colors = np.linspace(0, 1, X.shape[0])
	mymap = plt.get_cmap(colormap_name)

	for i in range(X.shape[0]):
		x_coord = X[i, 0]
		y_coord = X[i, 1]
		coords_x.append(x_coord)
		coords_y.append(y_coord)

	plt.scatter(np.array(coords_x), np.array(coords_y), s = 40, c = colors, edgecolors = 'None', cmap = mymap)
	plt.savefig(figure_name + "_" + colormap_name + ".jpg")

def plot_hypercolumns(X, net, label_map, frm_map, figure_name, hypercolumns_layers, encoding_func = None):
	hc_string = ''
	for layer in hypercolumns_layers:
		hc_string += layer
	X_pca = pca(X)
	X_tsne_pca = tsne_pca(X)
	plot_annotated_embedding(X_pca, label_map, frm_map,
		figure_name + '_'+ net +'_' + "Hypercolumn" + hc_string + '_pca', title = 'PCA - '+ net +' ' + " Hypercolumn "+ hc_string)
	plot_annotated_embedding(X_tsne_pca, label_map, frm_map,
		figure_name + '_'+ net +'_' + "Hypercolumn" + hc_string + '_tsne_pca', title = 't-SNE(PCA Input) - '+ net +' ' + " Hypercolumn " + hc_string)

def plot_all_layers(X, net, label_map, frm_map, figure_name, list_of_layers = constants.alex_net_layers, encoding_func = None):

	data = {}

	for layer in list_of_layers:
		print "----- Plotting layer " + str(layer) + " ---------"
		X_layer = X[layer]
		if encoding_func:
			X_layer = encoding_func(X_layer)
		X_pca = pca(X_layer, PC = 2)
		X_tsne_pca = tsne_pca(X_layer)
		X_grp = grp(X_layer, C = 2)
		plot_pylab_scatter(X_pca, label_map, frm_map,
			figure_name + '_'+ net +'_' + layer + '_PCA')
		plot_pylab_scatter(X_tsne_pca, label_map, frm_map,
			figure_name + '_'+ net +'_' + layer + '_t-SNE')
		plot_pylab_scatter(X_grp, label_map, frm_map,
			figure_name + '_'+ net +'_' + layer + '_GRP')
		data[layer] = [X_pca, X_tsne_pca, X_grp]
	pickle.dump(data, open(figure_name + "_dimred.p", "wb"))

def plot_all_layers_joint(X1, net, label_map_1, frm_map_1, X2, label_map_2, frm_map_2, figure_name, layers = constants.alex_net_layers,  encoding_func = None):
	num_X1_pts = X1[layers[0]].shape[0]

	for layer in layers:
		print "----- Plotting layer " + str(layer) + " ---------"
		X_layer1 = X1[layer]
		X_layer2 = X2[layer]
		if encoding_func is not None:
			X_layer1 = encoding_func(X_layer1)
			X_layer2 = encoding_func(X_layer2)
		X_joint = np.concatenate((X_layer1, X_layer2), axis = 0)
		X_pca = pca(X_joint)
		X_tsne = tsne(X_joint)

		name = figure_name + '_'+ net +'_' + layer
		plot_annotated_joint(X_pca, num_X1_pts, label_map_1, frm_map_1, label_map_2, frm_map_2,
			name + '_pca', title = 'PCA - ' + name)
		plot_annotated_joint(X_tsne, num_X1_pts, label_map_1, frm_map_1, label_map_2, frm_map_2,
			name + '_tsne', title = 't-SNE' + name)

def get_full_image_path(PATH_TO_DATA, frm_num):
	return PATH_TO_DATA + get_frame_fig_name(frm_num)

def vlad_experiment(X, list_of_K, list_of_PC, label_map, frm_map, figure_name, list_of_layers = constants.alex_net_layers):
	assert len(list_of_K) != 0
	assert len(list_of_PC) != 0

	for K in list_of_K:
		for PC in list_of_PC:
			for layer in list_of_layers:
				print("[VLAD] K = %3d  PC = %3d  layer = %s " % (K, PC, layer))

				X_layer = X[layer]
				X_vlad = encoding.encode_VLAD(X_layer, K, PC)

				X_vlad_pca = pca(X_vlad)
				X_vlad_tsne = tsne(X_vlad)

				name = figure_name + '_VLAD_' + 'K' + str(K) + '_PC'+ str(PC) + '_' + layer
				plot_annotated_embedding(X_vlad_pca, label_map, frm_map, name + '_pca', title = "PCA - " + name)
				plot_annotated_embedding(X_vlad_tsne, label_map, frm_map, name + '_tsne', title = "t-SNE - " + name)

def get_frame_fig_name(frm_num):
	"""
	Useful for parsing frames and loading into memory.
	"""
	if len(str(frm_num)) == 1:
		return "00000" + str(frm_num) + ".jpg"
	elif len(str(frm_num)) == 2:
		return "0000" + str(frm_num) + ".jpg"
	elif len(str(frm_num)) == 3:
		return "000" + str(frm_num) + ".jpg"
	elif len(str(frm_num)) == 4:
		return "00" + str(frm_num) + ".jpg"
	else:
		pass

def get_chronological_sequences(annotations):
	"""
	For all surgemes/regimes in annotations dict, returns segments, defined
	as (start, end) tuples in chronologically sorted order.
    """
	sequences = []
	for elem1 in annotations.values():
		for elem2 in elem1:
			sequences.append(elem2)

	sequences.sort(key = lambda x: x[0])
	return sequences

def hashcode():
	"""
	Returns hashcode to uniquely define a
    clustering trial run.
    """
	return str(random.randrange(1000, 10000))

def flatten(data):
	"""
	Flattens np array and returns as 1 * N array.
	"""
	return reshape(data.flatten())

def reshape(data):
	"""
	Reshapes any 1-D np array with shape (N,) to (1,N).
	"""
	return data.reshape(1, data.shape[0])

def dict_insert_list(key, value, dict):
	"""
    Used to maintain a dictionary of lists. For given key-value
    pair, function checks if key exists before inserting.
    """
	if key not in dict:
		dict[key] = [value,]
	else:
		curr_list = dict[key]
		curr_list.append(value)
		dict[key] = curr_list

def sys_copy(from_path, to_path):
	"""
    Executes copy function on filesystem
    given the input (from_path) and output (to_path) files to copy.
    """
	command = "cp " + from_path + " " + to_path
	os.system(command)

def dict_insert(key, value, data_dict, axis = 0):
	"""
	Inserts (key, value) pair into data_dict. Dictionary values are numpy arrays.
    """
	if key not in data_dict:
		data_dict[key] = value
	else:
		curr_value = data_dict[key]
		curr_value = np.concatenate((curr_value, value), axis = axis)
		data_dict[key] = curr_value

def safe_concatenate(X, W, axis = 0):
	"""
    Checks if X is None before concatenating W
    to X along specified axis (0 by default)
	"""
	if X is None:
		return W
	else:
		return np.concatenate((X, W), axis = axis)

def sample_matrix(matrix, sampling_rate = 1):
	"""
	Uniform sampling of matrix.
	Input: (N * d) matrix and sampling_rate
	Output: Sampled ((N/sampling_rate) * d) matrix
    """
	return matrix[::sampling_rate]

def nsf(num, n = 3):
    """
    n-Significant Figures
    """
    numstr = ("{0:.%ie}" % (n-1)).format(num)
    return float(numstr)

def print_and_write(content, file):
	"""
    Prints to STDOUT and to given file
	"""
	# print content
	file.write(content)

def print_and_write_2(metric, mean, std, file):
	# print("\n%1.3f  %1.3f  %s\n" % (mean, std, metric))
	file.write("\n%1.3f  %1.3f  %s\n" % (mean, std, metric))

def parse_yaml(yaml_fname):
	config = yaml.load(open(yaml_fname, 'r'))
	return config

def label_convert_to_numbers(labels):
	"""
    Converts labels of some type (e.g. String "A0" "1->3", etc to numerals).
    Helpful for passing labels into sklearn metric functions.
	"""
	new_labels = []
	one2one_mapping = {}
	index = 0
	unique_labels = set(labels)
	for unique_label in unique_labels:
		one2one_mapping[unique_label] = index
		index += 1

	for elem in labels:
		new_labels.append(one2one_mapping[elem])
	assert len(new_labels) == len(labels)

	return new_labels

def make_transition_feature(matrix, temporal_window, index):
	"""
	Input: Matrix X with start index i and window t
	Output: N = np.array(X[i], X[i+1], ... X[i + t])
	"""
	result = None
	for i in range(temporal_window + 1):
		result = safe_concatenate(result, reshape(matrix[index + i]), axis = 1)
	return result

def only_X(W):
	"""
	Used for Partially-Observed (PO) cases. Given kinematic state represented by (x,y)
	coordinates, this function returns (x,).
	"""
	return W.T[:1].T

def quaternion2rotation(q):
	"""
	Transform a unit quaternion into its corresponding rotation matrix (to
	be applied on the right side).
	"""
	(x, y, z, w) = q
	xx2 = 2 * x * x
	yy2 = 2 * y * y
	zz2 = 2 * z * z
	xy2 = 2 * x * y
	wz2 = 2 * w * z
	zx2 = 2 * z * x
	wy2 = 2 * w * y
	yz2 = 2 * y * z
	wx2 = 2 * w * x
	rmat = np.empty((3, 3), float)
	rmat[0,0] = 1. - yy2 - zz2
	rmat[0,1] = xy2 - wz2
	rmat[0,2] = zx2 + wy2
	rmat[1,0] = xy2 + wz2
	rmat[1,1] = 1. - xx2 - zz2
	rmat[1,2] = yz2 - wx2
	rmat[2,0] = zx2 - wy2
	rmat[2,1] = yz2 + wx2
	rmat[2,2] = 1. - xx2 - yy2
	return rmat

def silhoutte_weighted(points, labels):
	"""
	Returns weighted silhoutte scores.
	"""
	silhoutte_scores = metrics.silhouette_samples(points, labels, metric='euclidean')
	map_label2score = {}
	N = points.shape[0]

	for i in range(N):
		score = silhoutte_scores[i]
		label = labels[i]
		dict_insert_list(label, score, map_label2score)
	list_weighted_scores = []

	for label in map_label2score.keys():
		list_weighted_scores.append(np.mean(map_label2score[label]))

	num_labels = len(map_label2score.keys())

	return np.sum(list_weighted_scores)/float(num_labels)

def binary_search(ranges, val):
	"""
    Performs binary search to find which segment [start:end]
    bin val falls within. Returns the segment.
	"""
	if len(ranges) == 1:
		return ranges[0]

	middle_index = len(ranges)/2
	middle = ranges[middle_index]

	left_ranges = ranges[:middle_index]

	right_ranges = ranges[middle_index + 1:]


	if middle[0] <= val <= middle[1]:
		return middle

	elif val < middle[0]:
		return binary_search(left_ranges, val)

	else:
		return binary_search(right_ranges, val)

def generate_list_of_demonstrations(config_file_name, include_camera = False):
	list_of_demonstrations = []
	with open(config_file_name, "rb") as f:
		for line in f:
			params = line.split()
			if len(params) != 0:
				if include_camera:
					list_of_demonstrations.append(params[0] + "_capture1")
					list_of_demonstrations.append(params[0] + "_capture2")
				else:
					list_of_demonstrations.append(params[0])					
	return list_of_demonstrations

def frame2surgeme_map_demonstration(PATH_TO_TRANSCRIPTION, demonstration):
	map_frame2surgeme = {}
	with open(PATH_TO_TRANSCRIPTION + demonstration + ".txt", "rb") as f:
		for line in f:
			line = line.split()
			start = int(line[0])
			end = int(line[1])
			surgeme = int(constants.map_surgeme_label[line[2]])
			i = start
			while i <= end:
				map_frame2surgeme[i] = surgeme
				i += 1
	return map_frame2surgeme

def get_all_frame2surgeme_maps(list_of_demonstrations):
	"""
	For each demonstration in list_of_demonstrations, function returns a map:
	[frame number] -> segment label
	"""

	map_frame2surgeme = {}

	for demonstration in list_of_demonstrations:

		map_frame2surgeme[demonstration] = frame2surgeme_map_demonstration(constants.PATH_TO_DATA +
			constants.TRANSCRIPTIONS_FOLDER, demonstration)

	return map_frame2surgeme

def convert_transcription_to_annotation(PATH_TO_TRANSCRIPTION, PATH_TO_ANNOTATION, demonstration):
	"""
	Converts transcription.txt file to annotations.p file containing a dictionary of surgeme labels
	"""

	segments = {}
	with open(PATH_TO_TRANSCRIPTION + demonstration + ".txt", "rb") as f:
		for line in f:
			line = line.split()
			start = int(line[0])
			end = int(line[1])
			segment_index = int(constants.map_surgeme_label[line[2]])
			if segment_index not in segments:
				segments[segment_index] = [(start, end),]
			else:
				curr_list = segments[segment_index]
				curr_list.append((start, end))
				segments[segment_index] = curr_list

	pickle.dump(segments, open(PATH_TO_ANNOTATION + demonstration + "_capture1.p", "wb"))
	pickle.dump(segments, open(PATH_TO_ANNOTATION + demonstration + "_capture2.p", "wb"))

def parse_annotations(list_of_demonstrations = None):
	"""
	Note that left and right cameres have same transcriptions/annotations
	"""
	if not list_of_demonstrations:
		list_of_demonstrations = generate_list_of_demonstrations(constants.PATH_TO_DATA + constants.CONFIG_FILE)
	for video in list_of_demonstrations:
		convert_transcription_to_annotation(constants.PATH_TO_DATA + constants.TRANSCRIPTIONS_FOLDER,
			constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER, video)

def get_start_end_annotations(PATH_TO_ANNOTATION):
	"""
	Given the annotation (pickle) file, function returns the start and end
	frames of the demonstration.
	"""
	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	list_of_end_start_pts = []

	for key in segments:
		list_of_keys = segments[key]
		for elem in list_of_keys:
			list_of_end_start_pts.append(elem[0])
			list_of_end_start_pts.append(elem[1])

	start = min(list_of_end_start_pts)
	end = max(list_of_end_start_pts)
	return start, end

def get_annotation_segments(PATH_TO_ANNOTATION):
	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	list_of_end_start_pts = []
	for key in segments:
		list_of_keys = segments[key]
		for elem in list_of_keys:
			list_of_end_start_pts.append(elem)
	return list_of_end_start_pts

def parse_kinematics(PATH_TO_KINEMATICS_DATA, PATH_TO_ANNOTATION, fname):
	"""
	Takes in PATH to kinematics data (a txt file) and outputs a N x 38 matrix,
	where N is the number of frames. There are 38 dimensions in the kinematic data

	39-41  (3) : Slave left tooltip xyz
	42-50  (9) : Slave left tooltip R
	51-53  (3) : Slave left tooltip trans_vel x', y', z'   
	54-56  (3) : Slave left tooltip rot_vel
	57     (1) : Slave left gripper angle 
	58-76  (19): Slave right
	"""
	start, end = get_start_end_annotations(PATH_TO_ANNOTATION)

	X = None
	if constants.SIMULATION:
		mat = scipy.io.loadmat(PATH_TO_KINEMATICS_DATA + fname)
		X = mat['x_traj']
		X = X.T
		# IPython.embed()
		# X = pickle.load(open(PATH_TO_KINEMATICS_DATA + fname + ".p", "rb"))
	elif constants.TASK_NAME in ["plane","lego"]:
		print "-- Parsing Kinematics for ", fname
		trajectory = pickle.load(open(PATH_TO_KINEMATICS_DATA + fname + ".p", "rb"))
		for frm in range(start, end + 1):
			try:
				traj_point = trajectory[frm - start]
			except IndexError as e:
				print e
				IPython.embed()
			# vector = list(traj_point.position[16:-12]) + list(traj_point.velocity[16:-12])
			X = safe_concatenate(X, reshape(traj_point))

	else:
		X = None
		all_lines = open(PATH_TO_KINEMATICS_DATA + fname + ".txt", "rb").readlines()
		i = start - 1
		if i < 0:
			i = 0 
		while i < end:
			traj = np.array(all_lines[i].split())
			slave = traj[constants.KINEMATICS_DIM:]
			X = safe_concatenate(X, reshape(slave))
			i += 1
	return X.astype(np.float)

def get_kinematic_features(demonstration):
	"""
	Marshalls request to format needed for parse_kinematics.
	"""
	return parse_kinematics(constants.PATH_TO_KINEMATICS, constants.PATH_TO_DATA
		+ constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA +".p", demonstration)

if __name__ == "__main__":
	colormap_name = "coolwarm"
	start = 0.20
	end = 0.30

	parse_annotations(list_of_demonstrations)

	# X = parse_kinematics(constants.PATH_TO_KINEMATICS, constants.PATH_TO_DATA + "annotations/0001_02_capture1.p", "0001_02.mat")

	# VGG
	data = pickle.load(open("VGG_dimred.p", "rb"))
	list_of_layers = ['conv4_1','conv4_3', 'conv5_1', 'conv5_3', 'pool5']
	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]
		plot_scatter_continous(X_pca[np.floor(X_pca.shape[0]*start):np.floor(X_pca.shape[0]*end),:], "VGG_PCA" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)
		plot_scatter_continous(X_tsne[np.floor(X_tsne.shape[0]*start):np.floor(X_tsne.shape[0]*end),:], "VGG_tSNE" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)

	# AlexNet
	data = pickle.load(open("AlexNet_dimred.p", "rb"))
	list_of_layers = ["conv3", "conv4", "pool5"]
	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]
		plot_scatter_continous(X_pca[np.floor(X_pca.shape[0]*start):np.floor(X_pca.shape[0]*end),:], "AlexNet_PCA" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)
		plot_scatter_continous(X_tsne[np.floor(X_tsne.shape[0]*start):np.floor(X_tsne.shape[0]*end),:], "AlexNet_tSNE" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)