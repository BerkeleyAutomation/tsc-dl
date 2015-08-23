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
	else:
		bar_size = 10

	for elem in list_of_frms:
		list_of_start_end.append((elem - (bar_size/2), bar_size))
		list_of_colors.append(color)
	return list_of_start_end, tuple(list_of_colors)

def plot_broken_barh(demonstration, list_of_frms_1, list_of_frms_2, list_of_frms_3, list_of_frms_4, save_fname = None):
	"""
	Parameters:
	-----------
	demonstration: String name of demonstration without camera specification , e.g. "Suturing_E001"

	list_of_frms_[1,2,3,4]: List of changepoint frames for each of 4 different clustering experiments.
	Use this to compare manually vs. automatically generated transition points.

	* For now, the list_of_frms are constrained to 4 for visualization sanity sake.
	"""

	all_frms = list_of_frms_1 + list_of_frms_2 + list_of_frms_3 + list_of_frms_4
	time_cluster = mixture.GMM(n_components=25, covariance_type='full', n_iter=1000, thresh = 5e-5, min_covar = 0.001)
	X = np.array(all_frms)
	X = X.reshape(len(all_frms), 1)
	time_cluster.fit(X)
	Y = time_cluster.predict(X)
	
	list_of_elem = []
	for i in range(len(Y)):
		list_of_elem.append((Y[i], X[i]))
	list_of_elem = sorted(list_of_elem, key = lambda x:x[1][0] )

	for elem in list_of_elem:
		print elem

	dict_time_clusters = {}
	for elem in list_of_elem:
		utils.dict_insert_list(elem[0], elem[1], dict_time_clusters)

	list_time_clusters = []
	for cluster in dict_time_clusters.keys():
		cluster_frames = dict_time_clusters[cluster]
		min_frm = min(cluster_frames)
		max_frm = max(cluster_frames)

		list_time_clusters.append((min_frm[0], max_frm[0]))


	labels_automatic_0, colors_automatic_0 = setup_automatic_labels_2(list_time_clusters, "k")

	PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p"

	start, end = parser.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p")

	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	labels_manual, colors_manual = setup_manual_labels(segments)


	labels_automatic_1, colors_automatic_1 = setup_automatic_labels(list_of_frms_1, "k")
	labels_automatic_2, colors_automatic_2 = setup_automatic_labels(list_of_frms_2, "k")
	labels_automatic_3, colors_automatic_3 = setup_automatic_labels(list_of_frms_3, "k")
	labels_automatic_4, colors_automatic_4 = setup_automatic_labels(list_of_frms_4, "k")

	fig, ax = plt.subplots()

	ax.broken_barh(labels_manual, (25, 2), facecolors = colors_manual)
	ax.broken_barh(labels_automatic_0, (21, 2), facecolors = colors_automatic_0)
	ax.broken_barh(labels_automatic_1, (17, 2), facecolors = colors_automatic_1)
	ax.broken_barh(labels_automatic_2, (13, 2), facecolors = colors_automatic_2)
	ax.broken_barh(labels_automatic_3, (9, 2), facecolors = colors_automatic_3)
	ax.broken_barh(labels_automatic_4, (5, 2), facecolors = colors_automatic_4)

	ax.set_ylim(3,29)
	ax.set_xlim(0, end + 100) # Need to replace this with start and end frames
	ax.set_xlabel('Frame number')
	ax.set_yticks([6, 10, 14, 18, 22, 26])
	# ax.set_yticks([15,25,35, 45, 55])
	ax.set_yticklabels(['Automatic4','Automatic3','Automatic2', 'Automatic1','Time Clustering', 'Manual'])
	ax.grid(True)

	if save_fname:
		plt.savefig(save_fname)
	else:
		plt.show()
	pass

if __name__ == "__main__":
	#Needle passing examples
	# demonstration = "Needle_Passing_D001"	
	# list_of_frms_1 = [3679, 3772, 4465, 2170, 2215, 2233, 3154, 3169, 3889, 3904]
	# list_of_frms_2 = [3679, 3772, 4465, 3154, 3163, 3889, 3908]

	#Suturing Examples
	demonstration = "Suturing_E001"
	list_of_frms_1 = [2596 , 2746, 2950, 1513, 1702, 1783, 1087, 1954, 205, 2227, 2386] #6282_4_PCA
	list_of_frms_2 = [814, 1513, 205, 265, 289, 673, 778, 2386, 2596, 2746, 2950, 943, 1681, 1954] #5991_4_PCA
	list_of_frms_3 = [1663, 2227, 3091, 3100, 3301, 1954, 1372, 1513, 673, 211, 265, 289, 784, 808] #4916_4_PCA  
	list_of_frms_4 = [289, 265, 676, 1084, 1099, 1369, 1501, 1567] #6961_4_PCA

	plot_broken_barh(demonstration, list_of_frms_1, list_of_frms_2, list_of_frms_3, list_of_frms_4)
