"""
Make a "broken" horizontal bar plot, i.e., one with gaps
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import IPython
import pickle

import utils
import constants
import parser

from sklearn import mixture
from dtw import dtw

def get_ticks(manual_labels):
	"""
	Returns x-coordinates of tick marks from manual label transitions.
	"""
	ticks = []
	for elem in manual_labels[:-1]:
		ticks.append(elem[0] + elem[1])
	return ticks

def setup_manual_labels(segments):
	"""
	Takes in segment start,end in DeepMilestones annotations format
	and converts it to format for Mathplotlib broken_barh.
	"""
	list_of_start_end = []
	list_of_colors = []

	for key in segments.keys():
		color = constants.color_map[key]

		for elem in segments[key]:
			list_of_start_end.append((elem[0], elem[1] - elem[0]))
			list_of_colors.append(color)

	return list_of_start_end, tuple(list_of_colors)

def compute_dtw(time_sequence_1, time_sequence_2):
	dist, cost, path =  dtw(sorted(time_sequence_1), sorted(time_sequence_2),
		dist = lambda x, y: np.linalg.norm(x - y, ord=2))
	return dist


def setup_automatic_labels_2(list_of_frms, color):
	list_of_start_end = []
	list_of_colors = []
	for elem in list_of_frms:
		list_of_start_end.append((elem[0], elem[1] - elem[0]))
		list_of_colors.append(color)
	return list_of_start_end, tuple(list_of_colors)

def setup_automatic_labels(list_of_frms, color):
	"""
	Takes in changepoints and converts it to format
	for Mathplotlib broken_barh.
	"""
	list_of_start_end = []
	list_of_colors = []

	if constants.SIMULATION:
		bar_size = 0.5
	elif constants.TASK_NAME == "plane" or constants.TASK_NAME == "lego":
		bar_size = 2
	else:
		bar_size = 10


	for elem in list_of_frms:
		list_of_start_end.append((elem - (bar_size/2), bar_size))
		list_of_colors.append(color)
	return list_of_start_end, tuple(list_of_colors)


def get_time_clusters(data, T_COMPONENTS):
	# k-fold validation (Leave one out)
	numDemos = len(data.keys()) + 1
	sizeTestSet = numDemos - 1

	list_of_frms = []
	all_frms = []
	for key in data.keys():
		elem = data[key]
		list_of_frms.append(elem)
		all_frms += elem
	
	N_COMPONENTS = min(T_COMPONENTS, len(all_frms))
	time_cluster = mixture.GMM(n_components = N_COMPONENTS, covariance_type='full', n_iter = 50000, thresh = 5e-7)

	X = np.array(all_frms)
	X = X.reshape(len(all_frms), 1)
	time_cluster.fit(X)
	Y = time_cluster.predict(X)

	means = time_cluster.means_
	covars = time_cluster.covars_

	#dpgmm = mixture.DPGMM(n_components = numDemos * 3, covariance_type='diag', n_iter = 10000, alpha = 1000, thresh= 1e-10)
	#dpgmm.fit(X)
	#Y = dpgmm.predict(X)
	#means = dpgmm.means_

	list_of_elem = []

	list_of_means = []
	
	for i in range(len(Y)):
		list_of_elem.append((Y[i], X[i], means[Y[i]][0], np.sqrt(covars[Y[i]][0][0])))
	
	list_of_elem = sorted(list_of_elem, key = lambda x:x[1][0] )	

	dict_time_clusters = {}
	for elem in list_of_elem:
		utils.dict_insert_list(elem[0], elem[1], dict_time_clusters)

	list_time_clusters = []
	list_time_clusters_noPrune = []
	for cluster in dict_time_clusters.keys():
		# get all frames in this cluster
		cluster_frames = dict_time_clusters[cluster]
		setClusterFrames = set([elem[0] for elem in cluster_frames])
		# test if frames in cluster are representative of the test set
		rep = []
		for id in range(sizeTestSet):
			elemSet = set(list_of_frms[id])
			commonElem = elemSet.intersection(setClusterFrames)
			id_in_cluster = 1. if len(commonElem) > 0 else 0.
			rep.append(id_in_cluster)

		pruneCluster = True if sum(rep)/sizeTestSet < constants.PRUNING_FACTOR_T else False

		min_frm = min(cluster_frames)
		max_frm = max(cluster_frames)
		
		mean = means[cluster][0]
		list_of_means.append(int(mean))
		std = np.sqrt(covars[cluster][0][0])
		
		leftFrame = max(min_frm[0], mean - std)
		rightFrame = min(max_frm[0], mean + std)

		list_time_clusters_noPrune.append((leftFrame, rightFrame))
		# keep for plotting is pruneFlag = 0
		if not(pruneCluster):			
			# list_time_clusters.append((min_frm[0], max_frm[0]))
			list_time_clusters.append((leftFrame, rightFrame))

	print "Number of Clusters pruned in Time Clustering: ",  len(list_time_clusters_noPrune) - len(list_time_clusters)

	labels_automatic, colors_automatic = setup_automatic_labels_2(list_time_clusters, "k")
	labels_automatic_0, colors_automatic_0 = setup_automatic_labels(list_of_frms[0], "k")
	return labels_automatic, colors_automatic, labels_automatic_0, colors_automatic_0, list_of_means, list_of_frms

def plot_broken_barh_all(demonstration, data_W, data_Z, data_ZW, save_fname = None, save_fname2 = None):
	"""
	Plots time-clusters for W, K, KW.
	"""

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p"
	start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p")
	length = end - start
	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))

	TASK = constants.TASK_NAME
	if (TASK in ["lego", "plane"]):
		end = end + 20
	elif (TASK in ["000", "010", "011", "100"]):
		end = end + 2
	else:
		end = end + 50

	labels_manual, colors_manual = setup_manual_labels(segments)
	labels_automatic_W, colors_automatic_W, labels_automatic_W_0, colors_automatic_W_0, means_W, list_of_frms_W = get_time_clusters(data_W, constants.N_COMPONENTS_TIME_W)
	labels_automatic_Z, colors_automatic_Z, labels_automatic_Z_0, colors_automatic_Z_0, means_Z, list_of_frms_Z = get_time_clusters(data_Z, constants.N_COMPONENTS_TIME_Z)
	labels_automatic_ZW, colors_automatic_ZW, labels_automatic_ZW_0, colors_automatic_ZW_0, means_ZW, list_of_frms_ZW = get_time_clusters(data_ZW, constants.N_COMPONENTS_TIME_ZW)

	fig, ax = plt.subplots()
	ax.broken_barh(labels_manual, (17, 2), facecolors = colors_manual)
	ax.broken_barh(labels_automatic_W, (13, 2), facecolors = colors_automatic_W)
	ax.broken_barh(labels_automatic_Z, (9, 2), facecolors = colors_automatic_Z)
	ax.broken_barh(labels_automatic_ZW, (5, 2), facecolors = colors_automatic_ZW)

	ax.set_ylim(3,21)
	ax.set_xlim(0, end)
	ax.set_xlabel('Frame number')
	ax.set_yticks([6, 10, 14, 18])
	ax.set_yticklabels(['ZW','Z','W', 'Manual'])

	if save_fname:
		plt.savefig(save_fname)
	else:
		plt.show()
	pass

	fig, ax = plt.subplots()
	ax.broken_barh(labels_manual, (17, 2), facecolors = colors_manual)
	ax.broken_barh(labels_automatic_W_0, (13, 2), facecolors = colors_automatic_W_0)
	ax.broken_barh(labels_automatic_Z_0, (9, 2), facecolors = colors_automatic_Z_0)
	ax.broken_barh(labels_automatic_ZW_0, (5, 2), facecolors = colors_automatic_ZW_0)

	ax.set_ylim(3,21)
	ax.set_xlim(0, end)
	ax.set_xlabel('Frame number')
	ax.set_yticks([6, 10, 14, 18])
	ax.set_yticklabels(['ZW_0','Z_0','W_0', 'Manual'])

	if save_fname2:
		plt.savefig(save_fname2)
	else:
		plt.show()
	pass

	time_sequence_1 = [elem[0] + elem[1] for elem in labels_manual]

	time_sequence_2 = means_W
	dtw_score_W = compute_dtw(time_sequence_1, time_sequence_2)

	time_sequence_2 = means_Z
	dtw_score_Z = compute_dtw(time_sequence_1, time_sequence_2)

	time_sequence_2 = means_ZW
	dtw_score_ZW = compute_dtw(time_sequence_1, time_sequence_2)

	dtw_score_W_normalized = dtw_score_W/float(length) * 100
	dtw_score_Z_normalized = dtw_score_Z/float(length) * 100
	dtw_score_ZW_normalized = dtw_score_ZW/float(length) * 100

	return dtw_score_W, dtw_score_Z, dtw_score_ZW, dtw_score_W_normalized, dtw_score_Z_normalized, dtw_score_ZW_normalized, length

def preprocess_labels(list_of_labels):
	processed_list_of_labels = []
	for elem in list_of_labels:
		if elem[1] < 4.0:
			processed_list_of_labels.append((elem[0], elem[1] + 4.0))
		else:
			processed_list_of_labels.append(elem)
	return processed_list_of_labels

def plot_broken_barh_from_pickle(demonstration, output_fname, labels_manual, colors_manual, labels_automatic_W,
	colors_automatic_W, labels_automatic_Z, colors_automatic_Z, labels_automatic_ZW, colors_automatic_ZW):

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p"
	start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p")
	length = end - start

	fig, ax = plt.subplots()

	# Plot 1) Manual 2) Time clusters
	ax.broken_barh(labels_manual, (17, 2), facecolors = colors_manual)
	ax.broken_barh(preprocess_labels(labels_automatic_W), (13, 2), facecolors = colors_automatic_W)
	ax.broken_barh(preprocess_labels(labels_automatic_Z), (9, 2), facecolors = colors_automatic_Z)
	ax.broken_barh(preprocess_labels(labels_automatic_ZW), (5, 2), facecolors = colors_automatic_ZW)

	TASK = constants.TASK_NAME
	if (TASK in ["lego", "plane"]):
		end = end + 20
	elif (TASK in ["000", "010", "011", "100"]):
		end = end + 10
	else:
		end = end + 50

	ticks = get_ticks(labels_manual)
	ax.set_ylim(3,21)
	ax.set_xlim(0, end)
	ax.set_xlabel('Frame number')
	ax.set_yticks([6, 10, 14, 18])
	ax.set_yticklabels(['Both (k + z)','Visual (z)','Kinematics (k)', 'Manual'])

	if output_fname:
		plt.savefig(output_fname)
	else:
		plt.show()
	pass

def plot_broken_barh(demonstration, data, save_fname = None, T = 10):
	"""
	Parameters:
	-----------
	demonstration: String name of demonstration without camera specification , e.g. "Suturing_E001"

	list_of_frms_[1,2,3,4]: List of changepoint frames for each of 4 different clustering experiments.
	Use this to compare manually vs. automatically generated transition points.

	* For now, the list_of_frms are constrained to 4 for visualization sanity sake.
	"""

	numDemos = min(5, len(data.keys()) + 1)
	sizeTestSet = numDemos - 1

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p"
	start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p")
	length = end - start
	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))

	fig, ax = plt.subplots()
	# Generate labels for 1) Manual 2) Time clusters

	
	labels_automatic_0, colors_automatic_0, labels_automatic_1, colors_automatic_1, means, list_of_frms = get_time_clusters(data, T)

	labels_manual, colors_manual = setup_manual_labels(segments)

	# Plot 1) Manual 2) Time clusters
	ax.broken_barh(labels_manual, (25, 2), facecolors = colors_manual)
	ax.broken_barh(labels_automatic_0, (21, 2), facecolors = colors_automatic_0)

	list_of_plot_ranges = [(17, 2), (13, 2), (9, 2), (5, 2)]

	for i in range(min(sizeTestSet, 4)):
		labels_automatic, colors_automatic = setup_automatic_labels(list_of_frms[i], "k")
		ax.broken_barh(labels_automatic, list_of_plot_ranges[i], facecolors = colors_automatic)

	TASK = constants.TASK_NAME
	if (TASK in ["lego", "plane"]):
		end = end + 20
	elif (TASK in ["000", "010", "011", "100"]):
		end = end + 10
	else:
		end = end + 50

	ticks = get_ticks(labels_manual)
	ax.set_ylim(3,29)
	ax.set_xlim(0, end)
	ax.set_xlabel('Frame number')
	# ax.set_xticks(ticks)
	ax.set_yticks([6, 10, 14, 18, 22, 26])
	ax.set_yticklabels(['Automatic4','Automatic3','Automatic2', 'Automatic1','Time Clustering', 'Manual'])

	if save_fname:
		plt.savefig(save_fname)
	else:
		plt.show()
	pass

	time_sequence_1 = [elem[0] + elem[1] for elem in labels_manual ]
	time_sequence_2 = means

	dtw_score = compute_dtw(time_sequence_1, time_sequence_2)
	normalized_dtw_score = dtw_score/float(length) * 100

	return dtw_score, normalized_dtw_score, length, labels_manual, colors_manual, labels_automatic_0, colors_automatic_0
