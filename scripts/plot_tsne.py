#!/usr/bin/env python

# Script to plot t-SNE visualizations

import numpy as np
import matplotlib.pyplot as plt
import pickle
import IPython
import sys
import cv2
import matplotlib.image as mpimg
from sklearn import (decomposition, preprocessing, manifold)

import utils
import constants

PATH_TO_FIGURE = "../tsne_plots/"
colormap_name = "coolwarm" # We found this to be the best!

start = 0.00 # Suturing_E001
end = 1.0 # Suturing_E001

def generate_SIFT():
	data = pickle.load(open("sift_features/SIFT_plane_9_1.p", "rb"))
	X_pca = utils.pca(data, PC = 2)
	X_tsne = utils.tsne(data)
	data_dimred = [X_pca, X_tsne]
	pickle.dump(data_dimred, open("SIFT_plane_9_dimred.p", "wb"))

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

	X_pca = utils.pca(X, PC = 2)
	X_tsne = utils.tsne(X)
	data_dimred = [X_pca, X_tsne]
	pickle.dump(X_tsne, open("raw_pixel_" + demonstration + "_dimred.p", "wb"))

def plot_scatter_continous(X, figure_name, title = None, colormap_name = "Accent", 
	changepoints = None, labels = None, plotter = None, end_frame = 1000000, interactive_mode = False):

	if plotter == None:
		plotter = plt
		plt.figure()

	start_frame_chgpts = np.floor(X.shape[0] * start)
	end_frame_chgpts = np.floor(X.shape[0] * end)

	X = X[start_frame_chgpts:end_frame_chgpts,:]

	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	coords_x = []
	coords_y = []
	colors = np.linspace(0, 1, X.shape[0])
	mymap = plt.get_cmap(colormap_name)

	for i in range(X.shape[0]):
		if i <= end_frame:
			x_coord = X[i, 0]
			y_coord = X[i, 1]
			coords_x.append(x_coord)
			coords_y.append(y_coord)

	if interactive_mode:
		plotter.scatter(np.array(coords_x), np.array(coords_y), s = 40, color = 'm')
	else:
		plotter.scatter(np.array(coords_x), np.array(coords_y), s = 40, c = colors, edgecolors = 'None', cmap = mymap)

	# Plotting changepoints on the scatter plot
	if changepoints:
		changepoints_x = []
		changepoints_y = []

		for frame_number in changepoints:
			if (frame_number < end_frame_chgpts) and (frame_number > start_frame_chgpts) and (frame_number <= end_frame):
				x_coord = X[frame_number - int(start_frame_chgpts), 0]
				y_coord = X[frame_number - int(start_frame_chgpts), 1]
				changepoints_x.append(x_coord)
				changepoints_y.append(y_coord)
		if interactive_mode:
			plotter.scatter(np.array(changepoints_x), np.array(changepoints_y), s = 100, marker = "^", color = None, edgecolors = 'k')
		else:
			plotter.scatter(np.array(changepoints_x), np.array(changepoints_y), s = 60, marker = "^", color="k", edgecolors = 'None')

	#Plotting labels
	if labels:
		manual_label_x = []
		manual_label_y = []
		for frame_number in labels:
			if (frame_number < end_frame_chgpts) and (frame_number > start_frame_chgpts) and (frame_number <= end_frame):
				x_coord = X[frame_number - int(start_frame_chgpts), 0]
				y_coord = X[frame_number - int(start_frame_chgpts), 1]
				manual_label_x.append(x_coord)
				manual_label_y.append(y_coord)
		if interactive_mode:
			plotter.scatter(np.array(manual_label_x), np.array(manual_label_y), s = 100, color = 'c')
		else:
			plotter.scatter(np.array(manual_label_x), np.array(manual_label_y), s = 80, facecolors='none', edgecolors = 'g')
	plt.savefig(PATH_TO_FIGURE + figure_name + "_" + colormap_name + ".jpg")

def plot_VGG(data, demonstration, changepoints = None, plotter = None, labels = None, end_frame = 1000000, interactive_mode = False, plot_pca = False, plot_tsne = False):

	# list_of_layers = ['conv5_3', 'conv4_3', 'pool5']

	list_of_layers = ['conv5_3']

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))

	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]

		X_pca = utils.sample_matrix(X_pca, 2)
		X_tsne = utils.sample_matrix(X_tsne, 2)

		print "VGG", layer, X_tsne.shape
		if plot_pca:
			plot_scatter_continous(X_pca, "plot_VGG_PCA" + layer+ "_" + str(start) + "_"+str(end),
				colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)
		if plot_tsne:
			if interactive_mode:
				plot_scatter_continous(X_tsne, "plot_VGG_tSNE" + layer+ "_" + str(start) + "_"+str(end),
					colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)
			else:
				plot_scatter_continous(X_tsne, "plot_VGG_tSNE" + layer+ "_" + str(end_frame),
						colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)

def plot_AlexNet(data, demonstration, changepoints = None, plotter = None, labels = None, end_frame = 1000000, interactive_mode = False, plot_pca = False, plot_tsne = False):

	# list_of_layers = ["conv3", "conv4", "pool5"]

	list_of_layers = ["pool5"]

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))

	for layer in list_of_layers:
		[X_pca, X_tsne, X_grp] = data[layer]

		X_pca = utils.sample_matrix(X_pca, 2)
		X_tsne = utils.sample_matrix(X_tsne, 2)

		print "AlexNet", layer, X_tsne.shape

		if plot_pca:
			plot_scatter_continous(X_pca, "plot_AlexNet_PCA" + layer+ "_" + str(start)+"_"+str(end),
				colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)
		if plot_tsne:
			if interactive_mode:
				plot_scatter_continous(X_tsne, "plot_AlexNet_tSNE" + layer + "_" + str(end_frame),
					colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)
			else:
				plot_scatter_continous(X_tsne, "plot_AlexNet_tSNE" + layer + "_" + str(start) + "_" + str(end), colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)

def plot_raw_image_pixels(data, demonstration, changepoints = None, plotter = None, labels = None, end_frame = 1000000, interactive_mode = False, plot_pca = False, plot_tsne = False):

	[X_pca, X_tsne] = data

	print "Raw Pixels", X_tsne.shape

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))

	if plot_pca:
		plot_scatter_continous(X_pca, "plot_raw_pixels_PCA" + "_" + str(start) + "_" + str(end),
			colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)

	if plot_tsne:
		plot_scatter_continous(X_tsne, "plot_raw_pixels_tSNE" + "_" + str(start) + "_" + str(end),
			colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)

def plot_SIFT(data, demonstration, changepoints = None, plotter = None, labels = None, end_frame = 1000000, interactive_mode = False, plot_pca = False, plot_tsne = False):

	[X_pca, X_tsne] = data
	X_pca = utils.sample_matrix(X_pca, 6)
	X_tsne = utils.sample_matrix(X_tsne, 6)

	print "SIFT", X_tsne.shape

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	# labels = utils.get_chronological_sequences(annotations)

	if plot_pca:
		plot_scatter_continous(X_pca, "plot_SIFT_PCA" + "_" + str(start) + "_" + str(end),
			colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter)
	if plot_tsne:
		plot_scatter_continous(X_tsne, "plot_SIFT_tSNE" + "_" + str(start) + "_" + str(end),
			colormap_name = colormap_name, changepoints = changepoints, labels = labels, plotter = plotter, end_frame = end_frame, interactive_mode = interactive_mode)

def plot_all_same_figure():
	changepoints = pickle.load(open("pickle_files/Suturing_E001_changepoints_Z1.p", "rb"))
	jackknife_index = 1
	demonstration = "Suturing_E001"

	f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

	# AlexNet
	data = pickle.load(open("pickle_files/Suturing_E001_AlexNet_dimred.p", "rb"))
	ax3.set_title('AlexNet')
	plot_AlexNet(data, changepoints[jackknife_index], demonstration, plotter = ax3)

	# SIFT
	data = pickle.load(open("pickle_files/Suturing_E001_SIFT_dimred.p", "rb"))
	ax2.set_title('SIFT')
	plot_SIFT(data, changepoints[jackknife_index], demonstration, plotter = ax2)

	# Raw Image Pixels
	data = pickle.load(open("pickle_files/Suturing_E001_raw_pixel_dimred.p", "rb"))
	ax1.set_title('Raw Image Pixels')
	plot_raw_image_pixels(data, changepoints[jackknife_index], demonstration, plotter = ax1)

	plt.show()

def plot_all():
	# Suturing_E001
	# changepoints = pickle.load(open("pickle_files/Suturing_E001_changepoints_Z1.p", "rb"))
	# jackknife_index = 1
	demonstration = "Suturing_E001"

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	start_d, end_d = utils.get_start_end_annotations(PATH_TO_ANNOTATION)

	# changepoints = changepoints[jackknife_index]
	changepoints_processed = []

	results = pickle.load(open("pickle_files/Suturing_E001_changepoints_Z2.p", "rb"))
	jackknife_index = 3
	demonstration = "Suturing_E001"

	changepoints = results[demonstration]['changepoints'][jackknife_index]
	labels = results[demonstration]['plot_labels_automatic']

	for elem in changepoints:
		changepoints_processed.append((elem[0] - start_d)/6)

	labels_processed = []
	for elem in labels:
		labels_processed.append((elem[0] - start_d)/6)

	# VGG
	data = pickle.load(open("pickle_files/Suturing_E001_VGG_dimred.p", "rb"))
	plot_VGG(data, demonstration, changepoints = None, labels = None, plot_tsne = True)

	# AlexNet
	data = pickle.load(open("pickle_files/Suturing_E001_AlexNet_dimred.p", "rb"))
	plot_AlexNet(data, demonstration, changepoints = None, labels = None, plot_tsne = True)

	# Raw Pixels
	data = pickle.load(open("pickle_files/Suturing_E001_raw_pixel_dimred.p", "rb"))
	plot_raw_image_pixels(data, demonstration, changepoints = None, labels = None, plot_tsne = True)

	# SIFT
	data = pickle.load(open("pickle_files/Suturing_E001_SIFT_dimred.p", "rb"))
	plot_SIFT(data, demonstration, changepoints = None, labels = None, plot_tsne = True)

def plot_interactive():
	results = pickle.load(open("pickle_files/Suturing_E001_changepoints_Z2.p", "rb"))
	jackknife_index = 3
	demonstration = "Suturing_E001"

	changepoints = results[demonstration]['changepoints'][jackknife_index]
	labels = results[demonstration]['plot_labels_automatic']

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
	start_d, end_d = utils.get_start_end_annotations(PATH_TO_ANNOTATION)

	changepoints_processed = []
	for elem in changepoints:
		changepoints_processed.append((elem[0] - start_d)/12)

	labels_processed = []
	for elem in labels:
		labels_processed.append((elem[0] - start_d)/12)

	total_frames = [int(elem[0]) for elem in changepoints]
	total_frames += [int(elem[0]) for elem in labels]
	total_frames.sort()

	for end_frame in total_frames:

		# Frame
		ax1 = plt.subplot(121)
		PATH_TO_FIGURE = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + str(constants.CAMERA) + "/"
		im = mpimg.imread(utils.get_full_image_path(PATH_TO_FIGURE, end_frame))
		ax1.set_title(str(end_frame))
		ax1.xaxis.set_visible(False)
		ax1.yaxis.set_visible(False)
		ax1.imshow(im)

		# AlexNet
		ax2 = plt.subplot(122)
		data = pickle.load(open("pickle_files/Suturing_E001_AlexNet_dimred.p", "rb"))
		ax2.set_title('AlexNet')
		ax2.set_ylim([-0.1, 1.1])
		ax2.set_xlim([-0.1, 1.1])
		ax2.xaxis.set_visible(False)
		ax2.yaxis.set_visible(False)
		plot_AlexNet(data, demonstration, changepoints = changepoints_processed, plotter = ax2, labels = labels_processed, plot_tsne = True, end_frame = (end_frame - start_d)/12, interactive_mode = True)

if __name__ == "__main__":

	if PATH_TO_FIGURE is None:
		print "ERROR: Please specify path to pyplot savefig"
		sys.exit()

	# plot_all()
	plot_interactive()