#!/usr/bin/env python

import IPython
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
	random_projection, preprocessing, covariance, cluster, neighbors)
import random
import os
import yaml
from decimal import Decimal

import encoding
import constants

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
	X_pca = decomposition.TruncatedSVD(n_components=PC).fit_transform(X_centered)
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
		plt.text(X[i, 0], X[i, 1], str(frm_num), color=constants.color_map[label_map[i]],
			fontdict = {'weight': 'bold', 'size': 10})
	plt.xticks([]), plt.yticks([])
 	if title is not None:
		plt.title(title)
	plt.savefig(constants.PATH_TO_SAVE_FIG + figure_name + '.jpg')

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
	for layer in list_of_layers:
		print "----- Plotting layer " + str(layer) + " ---------"
		X_layer = X[layer]
		if encoding_func:
			X_layer = encoding_func(X_layer)
		X_pca = pca(X_layer)
		X_tsne_pca = tsne_pca(X_layer)
		plot_annotated_embedding(X_pca, label_map, frm_map,
			figure_name + '_'+ net +'_' + layer + '_pca', title = 'PCA - '+ net +' ' + layer)
		plot_annotated_embedding(X_tsne_pca, label_map, frm_map,
			figure_name + '_'+ net +'_' + layer + '_tsne_pca', title = 't-SNE(PCA Input) - '+ net +' ' + layer)

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

def get_chronological_sequences(annotations_file):
	sequences = []
	for elem1 in annotations_file.values():
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
