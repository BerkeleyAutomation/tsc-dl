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

start = 0.00 # Suturing_E001
end = 1.0 # Suturing_E001

# start = 0.0 # plane_9
# end = 1.0 # plane_9

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
			if ((frm % 6) == 0):
				PATH_TO_IMAGE = utils.get_full_image_path(constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + constants.CAMERA + "/", frm)
				print demonstration, str(frm)
				img = utils.reshape(cv2.imread(PATH_TO_IMAGE).flatten())
				X = utils.safe_concatenate(X, img)

	X_pca = pca(X, PC = 2)
	# X_tsne = tsne(X)
	# data_dimred = [X_pca, X_tsne]
	pickle.dump(X_tsne, open("raw_pixel_" + demonstration + "_dimred.p", "wb"))

def plot_raw_image_pixels(data, chpts, demonstration):

	[X_pca, X_tsne] = data

	print "Raw Pixels", X_tsne.shape

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	manual_labels = utils.get_chronological_sequences(annotations)

	plot_scatter_continous(X_pca, "plot_raw_pixels_PCA" + "_" + str(start) + "_" + str(end),
		colormap_name = colormap_name, changepoints = chpts, manual_labels = None)
	plot_scatter_continous(X_tsne, "plot_raw_pixels_tSNE" + "_" + str(start) + "_" + str(end),
		colormap_name = colormap_name, changepoints = chpts, manual_labels = None)


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

def plot_scatter_continous(X, figure_name, title = None, colormap_name = "Accent", 
	changepoints = None, manual_labels = None):

	start_frame_chgpts = np.floor(X.shape[0] * start)
	end_frame_chgpts = np.floor(X.shape[0] * end)

	X = X[start_frame_chgpts:end_frame_chgpts,:]

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

	# Plotting changepoints on the scatter plot
	if changepoints:
		changepoints_x = []
		changepoints_y = []
		for elem in changepoints:
			frame_number = elem[0]/3

			if (frame_number < end_frame_chgpts) and (frame_number > start_frame_chgpts):
				x_coord = X[frame_number - int(start_frame_chgpts), 0]
				y_coord = X[frame_number - int(start_frame_chgpts), 1]
				changepoints_x.append(x_coord)
				changepoints_y.append(y_coord)

		plt.scatter(np.array(changepoints_x), np.array(changepoints_y), s = 60, marker = "^", color="k", edgecolors = 'None')

	#Plotting manual_labels
	if manual_labels:
		manual_label_x = []
		manual_label_y = []
		for elem in manual_labels[:-1]:
			frame_number = elem[1]/3

			if (frame_number < end_frame_chgpts) and (frame_number > start_frame_chgpts):
				x_coord = X[frame_number - int(start_frame_chgpts), 0]
				y_coord = X[frame_number - int(start_frame_chgpts), 1]
				manual_label_x.append(x_coord)
				manual_label_y.append(y_coord)

		plt.scatter(np.array(manual_label_x), np.array(manual_label_y), s = 80, facecolors='none', edgecolors = 'g')

	plt.savefig(PATH_TO_FIGURE + figure_name + "_" + colormap_name + ".jpg")


def plot_VGG(data, chpts, demonstration):

	list_of_layers = ['conv5_3', 'conv4_3', 'pool5']

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	manual_labels = utils.get_chronological_sequences(annotations)

	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]

		X_pca = sample_matrix(X_pca, 2)
		X_tsne = sample_matrix(X_tsne, 2)

		print "VGG", layer, X_tsne.shape

		plot_scatter_continous(X_pca, "plot_VGG_PCA" + layer+ "_" + str(start) + "_"+str(end),
			colormap_name = colormap_name, changepoints = chpts, manual_labels = None)
		plot_scatter_continous(X_tsne, "plot_VGG_tSNE" + layer+ "_" + str(start) + "_"+str(end),
			colormap_name = colormap_name, changepoints = chpts, manual_labels = None)

def plot_AlexNet(data, chpts, demonstration):

	list_of_layers = ["conv3", "conv4", "pool5"]

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	manual_labels = utils.get_chronological_sequences(annotations)

	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]

		X_pca = sample_matrix(X_pca, 2)
		X_tsne = sample_matrix(X_tsne, 2)

		print "AlexNet", layer, X_tsne.shape

		plot_scatter_continous(X_pca, "plot_AlexNet_PCA" + layer+ "_" + str(start)+"_"+str(end),
			colormap_name = colormap_name, changepoints = chpts, manual_labels = None)
		plot_scatter_continous(X_tsne, "plot_AlexNet_tSNE" + layer+ "_" + str(start)+"_"+str(end),
			colormap_name = colormap_name, changepoints = chpts, manual_labels = None)

def generate_SIFT():
	data = pickle.load(open("sift_features/SIFT_plane_9_1.p", "rb"))
	X_pca = pca(data, PC = 2)
	X_tsne = tsne(data)
	data_dimred = [X_pca, X_tsne]
	pickle.dump(data_dimred, open("SIFT_plane_9_dimred.p", "wb"))

# def plot_SIFT():
# 	data = pickle.load(open("SIFT_plane_9_dimred.p", "rb"))
# 	[X_pca, X_tsne] = data
# 	# X_pca = sample_matrix(X_pca, 3)
# 	# X_tsne = sample_matrix(X_tsne, 3)
# 	plot_scatter_continous(X_pca, "plot_SIFT_plane_9_PCA_" + str(start)+"_"+str(end), colormap_name = colormap_name)
# 	plot_scatter_continous(X_tsne, "plot_SIFT_plane_9_tSNE_" + str(start)+"_"+str(end), colormap_name = colormap_name)

def plot_SIFT(data, chpts, demonstration):

	[X_pca, X_tsne] = data
	X_pca = sample_matrix(X_pca, 6)
	X_tsne = sample_matrix(X_tsne, 6)

	print "SIFT", X_tsne.shape

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	manual_labels = utils.get_chronological_sequences(annotations)

	plot_scatter_continous(X_pca, "plot_SIFT_PCA" + "_" + str(start) + "_" + str(end),
		colormap_name = colormap_name, changepoints = chpts, manual_labels = None)
	plot_scatter_continous(X_tsne, "plot_SIFT_tSNE" + "_" + str(start) + "_" + str(end),
		colormap_name = colormap_name, changepoints = chpts, manual_labels = None)


if __name__ == "__main__":

	if PATH_TO_FIGURE is None:
		print "ERROR: Please specify path to pyplot savefig"
		sys.exit()

	# Suturing_E001
	data = pickle.load(open("Suturing_E001_VGG_dimred.p", "rb"))
	chpts = pickle.load(open("Suturing_E001_changepoints_Z1.p", "rb"))
	jackknife_index = 1
	demonstration = "Suturing_E001"

	# plane_9
	# data = pickle.load(open("plane_9_VGG_dimred.p", "rb"))
	# chpts = pickle.load(open("plane_9_changepoints_Z.p", "rb"))
	# jackknife_index = 1
	# demonstration = "plane_9"

	# # VGG
	plot_VGG(data, chpts[jackknife_index], demonstration)

	data = pickle.load(open("Suturing_E001_AlexNet_dimred.p", "rb"))
	# # AlexNet
	plot_AlexNet(data, chpts[jackknife_index], demonstration)

	data = pickle.load(open("raw_pixel_Suturing_E001_dimred.p", "rb"))

	plot_raw_image_pixels(data, chpts[jackknife_index], demonstration)

	data = pickle.load(open("Suturing_E001_SIFT_dimred.p", "rb"))

	plot_SIFT(data, chpts[jackknife_index], demonstration)