import IPython
import numpy as np
import matlab.engine as mateng
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection, preprocessing, covariance)
import argparse
import cv2

CAFFE_ROOT = '/home/animesh/caffe/'

color_map = {1:'b', 2:'g', 3:'r', 4:'c', 5: 'm', 6:'y', 7:'k', 8:'#4B0082', 9: '#9932CC', 10: '#E9967A', 11: '#800000', 12: '#008080'}

# E9967A is beige/dark salmon
# 4B0082 is Indigo
# 800000 is Maroon 
# 008080 IS Teal

alex_net_layers = ['input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']

PATH_TO_SAVE_FIG = '/home/animesh/DeepMilestones/plots/'

def pca(X):
	print("Computing PCA embedding")
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X_centered)
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

def parse_annotations_images(annotations_file, PATH_TO_DATA):
	index_map = {}
	label_map = {}
	X = None
	i = 0
	map_index_data = pickle.load(open(annotations_file, "rb"))
	for index in map_index_data:
		print "Parsing label " + str(index)
		segments = map_index_data[index]
		for seg in segments:
			j = seg[0]
			while j <= seg[1]:
				im = cv2.imread(PATH_TO_DATA + get_frame_fig_name(j))
				im = im.flatten()
				im = im.reshape(1, im.shape[0])
				print j
				if X is None:
					X = im
				else:
					X = np.concatenate((X, im), axis = 0)
				index_map[i] = j
				label_map[i] = index
				j += 1
				i += 1
	return X, label_map, index_map


# Plots 2 set of conv5b PCA vectors on same image
def plot_annotated_joint(X_joint, num_X1_pts, label_map_1, index_map_1, label_map_2, index_map_2, figure_name, title = None):
	x_min, x_max = np.min(X_joint, 0), np.max(X_joint, 0)
	X_joint = (X_joint - x_min) / (x_max - x_min)
	plt.figure()
	for i in range(X_joint.shape[0]):
		if i < num_X1_pts:
			frm_num = index_map_1[i]
			plt.text(X_joint[i, 0], X_joint[i, 1], 'x'+str(frm_num),
				color=color_map[label_map_1[i]], fontdict={'weight': 'bold', 'size': 9})
		else:
			frm_num = index_map_2[i - num_X1_pts]
			plt.text(X_joint[i, 0], X_joint[i, 1], 'o'+str(frm_num),
				color=color_map[label_map_2[i - num_X1_pts]], fontdict={'weight': 'bold', 'size': 9})
	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)
	plt.savefig(PATH_TO_SAVE_FIG + figure_name + '.jpg')

# Scale and visualize the embedding vectors
def plot_annotated_embedding(X, label_map, index_map, figure_name, title=None):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	plt.figure()
	for i in range(X.shape[0]):
		frm_num = index_map[i]
		plt.text(X[i, 0], X[i, 1], str(frm_num), color=color_map[label_map[i]], fontdict={'weight': 'bold', 'size': 10})
	plt.xticks([]), plt.yticks([])
 	if title is not None:
		plt.title(title)
	plt.savefig(PATH_TO_SAVE_FIG + figure_name + '.jpg')

def plot_all_layers(X, label_map, index_map, figure_name):
	for layer in alex_net_layers:
		print "----- Plotting layer " + str(layer) + " ---------"
		X_layer = X[layer]
		X_pca = pca(X_layer)
		X_tsne_pca = tsne_pca(X_layer)
		plot_annotated_embedding(X_pca, label_map, index_map,
			figure_name + '_AlexNet_' + layer + '_pca', title = 'PCA - AlexNet ' + layer)
		plot_annotated_embedding(X_tsne_pca, label_map, index_map,
			figure_name + '_AlexNet_' + layer + '_tsne_pca', title = 't-SNE(PCA Input) - AlexNet ' + layer)

def plot_all_layers_joint(X1, label_map_1, index_map_1, X2, label_map_2, index_map_2, figure_name):

	num_X1_pts = X1[alex_net_layers[0]].shape[0]

	for layer in alex_net_layers:
		print "----- Plotting layer " + str(layer) + " ---------"
		X_layer1 = X1[layer]
		X_layer2 = X2[layer]

		X_joint = np.concatenate((X_layer1, X_layer2), axis = 0)
		X_pca = pca(X_joint)
		X_tsne_pca = tsne_pca(X_joint)

		plot_annotated_joint(X_pca, num_X1_pts, label_map_1, index_map_1, label_map_2, index_map_2,
			figure_name + '_AlexNet_' + layer + '_pca', title = 'PCA - AlexNet ' + layer)
		plot_annotated_joint(X_tsne_pca, num_X1_pts, label_map_1, index_map_1, label_map_2, index_map_2,
			figure_name + '_AlexNet_' + layer + '_tsne_pca', title = 't-SNE(PCA Input) - AlexNet ' + layer)

def get_frame_fig_name(frm_num):
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