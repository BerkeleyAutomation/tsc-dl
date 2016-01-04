#!/usr/bin/env python

# Script to plot t-SNE visualizations

import numpy as np
import matplotlib.pyplot as plt
import pickle
import IPython
import sys
import utils
import constants
import cv2

from sklearn import (decomposition, preprocessing, manifold)

PATH_TO_FIGURE = "../tsne_plots/"
colormap_name = "coolwarm" # We found this to be the best!
start = 0.25
end = 0.75

def generate_raw_image_pixels(list_of_demonstrations):
	"""
	PCA and t-SNE on raw image pixels
    """

	# Design matrix of raw image pixels
	X = None

	for demonstration in list_of_demonstrations:
		print "Raw image pixels ", demonstration
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"

		start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)
		for frm in range(start, end + 1):
			if ((frm % 3) == 0):
				PATH_TO_IMAGE = utils.get_full_image_path(constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + constants.CAMERA + "/", frm)
				print demonstration, str(frm)
				img = utils.reshape(cv2.imread(PATH_TO_IMAGE).flatten())
				X = utils.safe_concatenate(X, img)

	X_pca = pca(X, PC = 2)
	X_tsne = tsne(X)
	data_dimred = [X_pca, X_tsne]
	pickle.dump(data_dimred, open("raw_pixel_dimred.p", "wb"))


def sample_matrix(matrix, sampling_rate = 1):
	"""
	Uniform sampling of matrix.
	Input: (N * d) matrix and sampling_rate
	Output: Sampled ((N/sampling_rate) * d) matrix
    """
	return matrix[::sampling_rate]

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

def tsne(X):
	print("Computing t-SNE embedding")
	scaler = preprocessing.StandardScaler().fit(X)
	X_centered = scaler.transform(X)
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	X_tsne = tsne.fit_transform(X_centered)
	return X_tsne

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
	plt.savefig(PATH_TO_FIGURE + figure_name + "_" + colormap_name + ".jpg")

def plot_VGG():
	data = pickle.load(open("VGG_dimred.p", "rb"))
	list_of_layers = ['conv5_3',]
	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]
		plot_scatter_continous(X_pca[np.floor(X_pca.shape[0]*start):np.floor(X_pca.shape[0]*end),:],
			"plot_VGG_PCA" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)
		plot_scatter_continous(X_tsne[np.floor(X_tsne.shape[0]*start):np.floor(X_tsne.shape[0]*end),:],
			"plot_VGG_tSNE" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)

def plot_AlexNet():
	data = pickle.load(open("AlexNet_dimred.p", "rb"))
	list_of_layers = ["conv3", "conv4", "pool5"]
	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]
		plot_scatter_continous(X_pca[np.floor(X_pca.shape[0]*start):np.floor(X_pca.shape[0]*end),:],
			"plot_AlexNet_PCA" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)
		plot_scatter_continous(X_tsne[np.floor(X_tsne.shape[0]*start):np.floor(X_tsne.shape[0]*end),:],
			"plot_AlexNet_tSNE" + layer+ "_" + str(start)+"_"+str(end), colormap_name = colormap_name)

def generate_SIFT():
	data = pickle.load(open("sift_features/SIFT_Suturing_E001_1.p", "rb"))
	X_pca = pca(data, PC = 2)
	X_tsne = tsne(data)
	data_dimred = [X_pca, X_tsne]
	pickle.dump(data_dimred, open("SIFT_dimred.p", "wb"))

def plot_SIFT():
	data = pickle.load(open("SIFT_dimred.p", "rb"))
	[X_pca, X_tsne] = data
	X_pca = sample_matrix(X_pca, 3)
	X_tsne = sample_matrix(X_tsne, 3)
	plot_scatter_continous(X_pca[np.floor(X_pca.shape[0]*start):np.floor(X_pca.shape[0]*end),:],
		"plot_SIFT_sampled_PCA_SIFT_" + str(start)+"_"+str(end), colormap_name = colormap_name)
	plot_scatter_continous(X_tsne[np.floor(X_tsne.shape[0]*start):np.floor(X_tsne.shape[0]*end),:],
		"plot_SIFT_sampled_tSNE_SIFT_" + str(start)+"_"+str(end), colormap_name = colormap_name)

if __name__ == "__main__":

	if PATH_TO_FIGURE is None:
		print "ERROR: Please specify path to pyplot savefig"
		sys.exit()

	list_of_demonstrations = ["Suturing_E001",]

	#SIFT
	# plot_SIFT()

	# # VGG
	# plot_VGG()

	# # AlexNet
	# plot_AlexNet()

	generate_raw_image_pixels(list_of_demonstrations)
