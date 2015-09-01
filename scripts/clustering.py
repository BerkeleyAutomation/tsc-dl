#!/usr/bin/env python

import pickle
import numpy as np
import random
import os
import argparse
import sys
import IPython
import itertools

import constants
import parser
import utils
import cluster_evaluation
import broken_barh

from sklearn import (mixture, preprocessing, neighbors, metrics, cross_decomposition)
from decimal import Decimal
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
mutual_info_score, homogeneity_score, completeness_score, recall_score, precision_score)

PATH_TO_FEATURES = constants.PATH_TO_DATA + constants.PROC_FEATURES_FOLDER

class MilestonesClustering():
	def __init__(self, DEBUG, list_of_demonstrations, featfile, trialname):	
		self.list_of_demonstrations = list_of_demonstrations
		self.data_X = {}
		self.data_W = {}
		self.data_Z = {}
		self.data_X_size = {}
		self.data_N = {}

		self.file = None
		self.featfile = featfile
		self.metrics_picklefile = None

		self.change_pts = None
		self.change_pts_Z = None
		self.change_pts_W = None

		self.list_of_cp = []
		self.map_cp2frm = {}
		self.map_cp2demonstrations = {}
		self.map_cp2level1 = {}
		self.map_level12cp = {}
		self.map_cp2milestones = {}
		self.map_cp2surgemes = {}
		self.map_cp2surgemetransitions = {}
		self.l2_cluster_matrices = {}
		self.map_frm2surgeme = parser.get_all_frame2surgeme_maps(self.list_of_demonstrations)
		self.trial = utils.hashcode() + trialname
		self.cp_surgemes = []
		self.pruned_L1_clusters = []
		self.pruned_L2_clusters = []

		self.silhouette_scores = {}
		self.dunn_scores_1 = {}
		self.dunn_scores_2 = {}
		self.dunn_scores_3 = {}

		self.level2_dunn_1 = []
		self.level2_dunn_2 = []
		self.level2_dunn_3 = []
		self.level2_silhoutte = []

		self.label_based_scores_1 = {}
		self.label_based_scores_2 = {}

		self.sr = constants.SR
		self.representativeness = constants.PRUNING_FACTOR

		# Components for Mixture model at each level
		self.n_components_cp = constants.N_COMPONENTS_CP
		self.n_components_L1 = constants.N_COMPONENTS_L1
		self.n_components_L2 = constants.N_COMPONENTS_L2

		self.temporal_window = constants.TEMPORAL_WINDOW

	def loads_features(self):
		"""
		Loads the kinematic and/or visual features into self.data_X.
		"""
		self.data_X = pickle.load(open(PATH_TO_FEATURES + str(self.featfile),"rb"))

		IPython.embed()
		for demonstration in self.list_of_demonstrations:
			if demonstration not in self.data_X.keys():
				print "[ERROR] Missing demonstrations"
				sys.exit()

	def construct_features_visual(self):
		"""
		Independently loads/sets-up the kinematics in self.data_Z.
		"""
		data_X = pickle.load(open(PATH_TO_FEATURES + str(self.featfile),"rb"))
		for demonstration in self.list_of_demonstrations:
			X = data_X[demonstration]
			Z = None
			for i in range(len(X)):
				Z = utils.safe_concatenate(Z, utils.reshape(X[i][constants.KINEMATICS_DIM:]))
			assert Z.shape[0] == X.shape[0]

			self.data_Z[demonstration] = Z

	def construct_features_kinematics(self):
		"""
		Independently loads/sets-up the kinematics in self.data_W.
		"""
		for demonstration in self.list_of_demonstrations:
			self.data_W[demonstration] = utils.sample_matrix(parser.get_kinematic_features(demonstration), sampling_rate = self.sr)

	def loads_features_split(self):
		"""
		Independently loads/sets-up the kinematics and visual data, then
		concatenates to populate self.data_X with X vectors.
		"""
		self.construct_features_kinematics()
		self.construct_features_visual()

		for demonstration in self.list_of_demonstrations:
			W = self.data_W[demonstration]
			Z = self.data_W[demonstration]

			assert W.shape[0] == Z.shape[0]
			assert W.shape[1] == constants.KINEMATICS_DIM

			self.data_X[demonstration] = utils.safe_concatenate(W, Z, axis = 1)

	def generate_transition_features(self):
		for demonstration in self.list_of_demonstrations:

			X = self.data_X[demonstration]
			self.data_X_size[demonstration] = X.shape[1]
			T = X.shape[0]
			N = None
			for t in range(T - self.temporal_window):

				n_t = utils.make_transition_feature(X, self.temporal_window, t)
				N = utils.safe_concatenate(N, n_t)

			self.data_N[demonstration] = N

	def generate_change_points_1(self):
		"""
		Generates changespoints by clustering within a demonstration.
		"""

		cp_index = 0

		for demonstration in self.list_of_demonstrations:

			N = self.data_N[demonstration]

			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', n_iter=1000, thresh = 5e-5)
			gmm.fit(N)
			Y = gmm.predict(N)
	
			self.save_cluster_metrics(N, Y, gmm.means_, 'cpts_' + demonstration, gmm)

			start, end = parser.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
				+ demonstration + "_" + constants.CAMERA + ".p")

			size_of_X = self.data_X_size[demonstration]

			for i in range(len(Y) - 1):

				if Y[i] != Y[i + 1]:
					change_pt = N[i][size_of_X:]
					self.append_cp_array(change_pt)
					self.map_cp2frm[cp_index] = start + i * self.sr
					self.map_cp2demonstrations[cp_index] = demonstration
					self.list_of_cp.append(cp_index)

					cp_index += 1

	def generate_change_points_2(self):
		"""
		Generates changespoints by clustering across demonstrations.
		"""
		cp_index = 0
		i = 0
		big_N = None
		map_index2demonstration = {}
		map_index2frm = {}
		size_of_X = self.data_X_size[self.list_of_demonstrations[0]]

		for demonstration in self.list_of_demonstrations:
			print demonstration
			N = self.data_N[demonstration]

			start, end = parser.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
				+ demonstration + "_" + constants.CAMERA + ".p")

			for j in range(N.shape[0]):
				map_index2demonstration[i] = demonstration
				map_index2frm[i] = start + j * self.sr
				i += 1

			big_N = utils.safe_concatenate(big_N, N)

		print "Generated big_N"

		if constants.REMOTE == 1:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', thresh = 0.01)
		elif constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', tol = 0.01)
		else:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full')
		gmm.fit(big_N)
		Y = gmm.predict(big_N)

		for w in range(len(Y) - 1):

			if Y[w] != Y[w + 1]:
				change_pt = big_N[w][size_of_X:]
				self.append_cp_array(change_pt)
				self.map_cp2frm[cp_index] = map_index2frm[w]
				self.map_cp2demonstrations[cp_index] = map_index2demonstration[w]
				self.list_of_cp.append(cp_index)

				cp_index += 1

		print "Done with generating change points"

	def append_cp_array(self, cp):
		if self.change_pts is None:
			self.change_pts = utils.reshape(cp)
			self.change_pts_W = utils.reshape(cp[:constants.KINEMATICS_DIM])
			self.change_pts_Z = utils.reshape(cp[constants.KINEMATICS_DIM:])

		else:
			try:
				self.change_pts = np.concatenate((self.change_pts, utils.reshape(cp)), axis = 0)
			except ValueError as e:
				print e
				sys.exit()
			self.change_pts_W = np.concatenate((self.change_pts_W, utils.reshape(cp[:constants.KINEMATICS_DIM])), axis = 0)
			self.change_pts_Z = np.concatenate((self.change_pts_Z, utils.reshape(cp[constants.KINEMATICS_DIM:])), axis = 0)


	def save_cluster_metrics(self, points, predictions, means, key, model, level2_mode = False):

		try:
			silhoutte = metrics.silhouette_score(points, predictions, metric='euclidean')
			self.silhouette_scores[key] = silhoutte
			if level2_mode:
				self.level2_silhoutte.append(silhoutte)

		except ValueError as e:
			pass

		dunn_scores = cluster_evaluation.dunn_index(points, predictions, means)

		if (dunn_scores[0] is not None) and (dunn_scores[1] is not None) and (dunn_scores[2] is not None):

			self.dunn_scores_1[key] = dunn_scores[0]
			self.dunn_scores_2[key] = dunn_scores[1]
			self.dunn_scores_3[key] = dunn_scores[2]

			if level2_mode:
				self.level2_dunn_1.append(dunn_scores[0])
				self.level2_dunn_2.append(dunn_scores[1])
				self.level2_dunn_3.append(dunn_scores[2])

	def cluster_changepoints_level1(self):

		# print "Level1 : Clustering changepoints in Z(t)"

		if constants.REMOTE == 1:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', n_iter=1000, thresh = 5e-5)
		elif constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', tol = 0.01)
		else:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full')

		gmm.fit(self.change_pts_Z)

		Y = gmm.predict(self.change_pts_Z)

		self.save_cluster_metrics(self.change_pts_Z, Y, gmm.means_, 'level1', gmm)

		for i in range(len(Y)):
			label = constants.alphabet_map[Y[i] + 1]
			self.map_cp2level1[i] = label
			utils.dict_insert_list(label, i, self.map_level12cp)

		self.generate_l2_cluster_matrices()

	def generate_l2_cluster_matrices(self):

		for key in sorted(self.map_level12cp.keys()):

			list_of_cp = self.map_level12cp[key]
			matrix = None

			for cp_index in list_of_cp:

				cp = utils.reshape(self.change_pts_W[cp_index])

				if matrix is None:
					matrix = cp
				else:
					matrix = np.concatenate((matrix, cp), axis = 0)

			self.l2_cluster_matrices[key] = matrix

	def cluster_changepoints_level2(self):

		# print "Level2 : Clustering changepoints in W(t)"

		mkdir_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial
		os.mkdir(mkdir_path)

		# To put frames of milestones
		os.mkdir(mkdir_path + "/" + "milestones")

		self.file = open(mkdir_path + "/" + self.trial + "clustering.txt", "wb")
		self.metrics_picklefile = mkdir_path + "/" + self.trial + "metrics.p"

		line = self.featfile + "\n\n"
		self.file.write(line)

		line = "L1 Cluster   L2 Cluster   Demonstration   Frame#  CP#   Surgeme\n"
		self.file.write(line)

		print "---Checking data representativeness ---"
		for key in sorted(self.map_level12cp.keys()):
			mkdir_l1_cluster = mkdir_path + "/" + key

			list_of_cp_key = self.map_level12cp[key]

			if self.check_pruning_condition(list_of_cp_key):
				continue

			os.mkdir(mkdir_l1_cluster)
		print "--- ---"

		for key in sorted(self.map_level12cp.keys()):
			matrix = self.l2_cluster_matrices[key]
			list_of_cp_key = self.map_level12cp[key]

			if self.check_pruning_condition(list_of_cp_key):
				self.pruned_L1_clusters.append(key)
				del self.map_level12cp[key]
				for pruned_cp in list_of_cp_key:
					print "Pruned: " + str(key) + " " + str(pruned_cp) + " " + str(self.map_cp2demonstrations[pruned_cp])
					self.list_of_cp.remove(pruned_cp)
				continue

			if constants.REMOTE == 1:
				gmm = mixture.GMM(n_components = self.n_components_L2, covariance_type='full', n_iter=1000, thresh = 5e-5)
			if constants.REMOTE == 2:
				gmm = mixture.GMM(n_components = self.n_components_L2, covariance_type='full', tol = 0.01)
			else:
				gmm = mixture.GMM(n_components = self.n_components_L2, covariance_type='full')

			try:
				gmm.fit(matrix)
			except ValueError as e:
				continue

			Y = gmm.predict(matrix)
			self.save_cluster_metrics(matrix, Y, gmm.means_, 'level2_' + str(key), gmm, level2_mode = True)

			for i in range(len(Y)):

				cp = list_of_cp_key[i]
				l1_cluster = key
				l2_cluster = Y[i]
				milestone = l1_cluster + str(l2_cluster)
				demonstration = self.map_cp2demonstrations[cp]
				try:
					frm = self.map_cp2frm[cp]
					surgeme = self.map_frm2surgeme[demonstration][frm]
				except KeyError as e:
					print e
					sys.exit()

				self.map_cp2milestones[cp] = milestone

				self.file.write("%s             %3d         %s   %3d   %3d    %3d\n" % (l1_cluster, l2_cluster, demonstration, frm, cp, surgeme))

				if constants.REMOTE == 0:
					self.copy_frames(demonstration, frm, str(l1_cluster), str(l2_cluster), surgeme)

		if constants.REMOTE == 0:
			self.copy_milestone_frames(matrix, list_of_cp_key, gmm)

	def copy_milestone_frames(self, matrix, list_of_cp_key, gmm):
		neigh = neighbors.KNeighborsClassifier(n_neighbors = 1)
		neigh.fit(matrix, list_of_cp_key)
		milestone_closest_cp = neigh.predict(gmm.means_)

		assert len(milestone_closest_cp) == self.n_components_L2

		for cp in milestone_closest_cp:
			demonstration = self.map_cp2demonstrations[cp]
			surgeme = self.map_frm2surgeme[demonstration][self.map_cp2frm[cp]]
			frm = utils.get_frame_fig_name(self.map_cp2frm[cp])

			from_path = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_" + constants.CAMERA + "/" + frm

			to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/milestones/" + self.map_cp2milestones[cp] + "_" + str(surgeme) + "_" + demonstration + "_" + frm

			utils.sys_copy(from_path, to_path)

	# Prune clusters which represent less than X% of the total demonstration data
	def check_pruning_condition(self, list_of_cp_key):
		demonstrations_in_cluster = [self.map_cp2demonstrations[cp] for cp in list_of_cp_key]

		num_demonstrations = len(set(demonstrations_in_cluster))
		num_total_demonstrations = len(self.list_of_demonstrations)
		data_representation = num_demonstrations / float(num_total_demonstrations)

		print data_representation, len(list_of_cp_key)

		return data_representation <= self.representativeness 

	def copy_frames(self, demonstration, frm, l1_cluster, l2_cluster, surgeme):

		from_path = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_" + constants.CAMERA + "/" + utils.get_frame_fig_name(frm)

		to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/" + l1_cluster + "/" + l2_cluster + "_" + str(surgeme) + "_" + demonstration + "_" + utils.get_frame_fig_name(frm)

		utils.sys_copy(from_path, to_path)

	def cluster_evaluation(self):

		for cp in self.list_of_cp:

			demonstration = self.map_cp2demonstrations[cp]
			frm = self.map_cp2frm[cp]

			curr_surgeme = self.map_frm2surgeme[demonstration][frm]
			self.map_cp2surgemes[cp] = curr_surgeme

			ranges = sorted(parser.get_annotation_segments(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
				+ demonstration + "_" + constants.CAMERA + ".p"))

			bin = utils.binary_search(ranges, frm)

			map_frm2surgeme_demonstration = self.map_frm2surgeme[demonstration]

			prev_end = bin[0] - 1
			next_start = bin[1] + 1
			prev_surgeme = map_frm2surgeme_demonstration[prev_end] if prev_end in map_frm2surgeme_demonstration else "start"
			next_surgeme = map_frm2surgeme_demonstration[next_start] if next_start in map_frm2surgeme_demonstration else "end"

			surgemetransition = None

			if abs(frm - (bin[0] - 1)) < abs(bin[1] + 1 - frm):
				surgemetransition = str(prev_surgeme) + "->" + str(curr_surgeme)
			else:
				surgemetransition = str(curr_surgeme) + "->" + str(next_surgeme)

			self.map_cp2surgemetransitions[cp] = surgemetransition

		self.cp_surgemes = set(self.map_cp2surgemes.values())


		# Initialize data structures
		table = {}
		for L1_cluster in self.map_level12cp.keys():
			new_dict = {}
			for surgeme in self.cp_surgemes:
				new_dict[surgeme] = 0
			table[L1_cluster] = new_dict

		surgeme_count = {}
		for surgeme in self.cp_surgemes:
			surgeme_count[surgeme] = 0

		for L1_cluster in self.map_level12cp.keys():
			list_of_cp_key = self.map_level12cp[L1_cluster]
			for cp in list_of_cp_key:
				surgeme = self.map_frm2surgeme[self.map_cp2demonstrations[cp]][self.map_cp2frm[cp]]
				surgeme_count[surgeme] += 1

				curr_dict = table[L1_cluster]
				curr_dict[surgeme] += 1
				table[L1_cluster] = curr_dict

		final_clusters = list(set(self.map_level12cp.keys()) - set(self.pruned_L1_clusters))

		confusion_matrix = "    "
		for surgeme in self.cp_surgemes:
			confusion_matrix = confusion_matrix + str(surgeme) + "     "

		# print confusion_matrix
		self.file.write('\n\n ---Confusion Matrix--- \n\n')
		self.file.write(confusion_matrix)

		confusion_matrix = ""
		for L1_cluster in final_clusters:
			confusion_matrix = confusion_matrix + "\n" + L1_cluster + "   "
			for surgeme in self.cp_surgemes:
				# confusion_matrix += str(float("{0:.2f}".format(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])))) + "   "
				confusion_matrix += str(round(Decimal(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])), 2)) + "   "
			confusion_matrix += '\n'

		# print confusion_matrix
		self.file.write(confusion_matrix)
		self.file.write("\n\n ---Surgeme Count--- \n\n")
		self.file.write(repr(surgeme_count))
		self.file.write("\n\n")

	def prepare_labels(self):
		labels_pred_1_ = []
		labels_pred_2_ = []
		labels_true_ = []

		for cp in self.list_of_cp:
			labels_true_.append(self.map_cp2surgemetransitions[cp])
			try:
				milestone_label = self.map_cp2milestones[cp]
			except KeyError:
				IPython.embed()
				print "Too Few elements inside cluster!!"
				sys.exit()
			labels_pred_1_.append(milestone_label)
			labels_pred_2_.append(list(milestone_label)[0])

		labels_pred_1 = utils.label_convert_to_numbers(labels_pred_1_)
		labels_pred_2 = utils.label_convert_to_numbers(labels_pred_2_)
		labels_true = utils.label_convert_to_numbers(labels_true_)

		assert len(labels_true_) == len(labels_pred_1_) == len(labels_pred_2_)
		assert len(labels_true) == len(labels_pred_1) == len(labels_pred_2)

		return labels_true, labels_pred_1, labels_pred_2

	def cluster_metrics(self):
		labels_true, labels_pred_1, labels_pred_2 = self.prepare_labels()

		metric_funcs = [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, 
		mutual_info_score, homogeneity_score, completeness_score]

		# ------ Label-based metrics ------
		self.file.write("\n\nPred = L2 Milestones       Pred= L1 Labels     Metric\n\n")

		for metric in metric_funcs:

			try:
				score_1 = utils.nsf(metric(labels_true, labels_pred_1))
				score_2 = utils.nsf(metric(labels_true, labels_pred_2))

			except ValueError as e:
				print e
				print "Pruning error"
				sys.exit()
			key =  repr(metric).split()[1]
			self.label_based_scores_1[key] = score_1
			self.label_based_scores_2[key] = score_2

			self.file.write("%3.3f        %3.3f        %s\n" % (score_1, score_2, key))

		# ------ Precision & Recall ------
		try:
			for ave in ["micro", "macro", "weighted"]:
				key = "precision_" + ave
				score_1 = utils.nsf(precision_score(labels_true, labels_pred_1, average = ave))
				score_2 = utils.nsf(precision_score(labels_true, labels_pred_2, average = ave))

				self.label_based_scores_1[key] = score_1
				self.label_based_scores_2[key] = score_2

				self.file.write("%3.3f        %3.3f        %s\n" % (score_1, score_2, key))

				key = "recall_" + ave
				score_1 = utils.nsf(recall_score(labels_true, labels_pred_1, average = ave))
				score_2 = utils.nsf(recall_score(labels_true, labels_pred_2, average = ave))

				self.label_based_scores_1[key] = score_1
				self.label_based_scores_2[key] = score_2

				self.file.write("%3.3f        %3.3f        %s\n" % (score_1, score_2, key))

		except ValueError as e:
			print e
			print "Pruning error"
			sys.exit()

		self.file.write("\nSilhouette Scores\n")

		# ------ Silhouette Scores ------
		for layer in sorted(self.silhouette_scores):
			score = self.silhouette_scores[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		self.file.write("\nDunn Scores1\n")

		# ------ Dunn Scores ------
		for layer in sorted(self.dunn_scores_1):
			score = self.dunn_scores_1[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		self.file.write("\nDunn Scores2\n")

		# ------ Dunn Scores ------
		for layer in sorted(self.dunn_scores_2):
			score = self.dunn_scores_2[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		self.file.write("\nDunn Scores3\n")

		# ------ Dunn Scores ------
		for layer in sorted(self.dunn_scores_3):
			score = self.dunn_scores_3[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))


		# ------ Visualizing changepoints on broken barh ------
		viz = {}

		for cp in self.list_of_cp:
			utils.dict_insert_list(self.map_cp2demonstrations[cp], self.map_cp2frm[cp], viz)

		data = [self.label_based_scores_1, self.label_based_scores_2,
		self.silhouette_scores, self.dunn_scores_1, self.dunn_scores_2, self.dunn_scores_3,
		self.level2_silhoutte, self.level2_dunn_1, self.level2_dunn_2, self.level2_dunn_3, viz]

		pickle.dump(data, open(self.metrics_picklefile, "wb"))

		return data

	def clean_up(self):
		self.file.close()

	def do_everything(self):

		# self.loads_features()
		self.loads_features_split()

		self.generate_transition_features()

		self.generate_change_points_2()

		self.cluster_changepoints_level1()

		self.cluster_changepoints_level2()

		self.cluster_evaluation()

		data = self.cluster_metrics()

		self.clean_up()

		return data

def get_list_of_demo_combinations(list_of_demonstrations):
	iterator = itertools.combinations(list_of_demonstrations, len(list_of_demonstrations) - 1)
	demo_combinations = []

	while (1):
		try:
			demo_combinations.append(iterator.next())
		except StopIteration as e:
			break

	return demo_combinations

def post_evaluation(metrics, filename, list_of_demonstrations, feat_fname):

	mutual_information_1 = []
	normalized_mutual_information_1 = []
	adjusted_mutual_information_1 = []
	homogeneity_1 = []
	precision_1_micro = []
	recall_1_micro = []
	precision_1_macro = []
	recall_1_macro = []
	precision_1_weighted = []
	recall_1_weighted = []

	mutual_information_2 = []
	normalized_mutual_information_2 = []
	adjusted_mutual_information_2 = []
	homogeneity_2 = []
	precision_2_micro = []
	recall_2_micro = []
	precision_2_macro = []
	recall_2_macro = []
	precision_2_weighted = []
	recall_2_weighted = []

	silhoutte_level_1 = []
	dunn1_level_1 = []
	dunn2_level_1 = []
	dunn3_level_1 = []

	silhoutte_level_2 = []
	dunn1_level_2 = []
	dunn2_level_2 = []
	dunn3_level_2 = []

	list_of_frms = {}

	for elem in metrics:
		precision_1_micro.append(elem[0]["precision_micro"])
		precision_2_micro.append(elem[1]["precision_micro"])

		precision_1_macro.append(elem[0]["precision_macro"])
		precision_2_macro.append(elem[1]["precision_macro"])

		precision_1_weighted.append(elem[0]["precision_weighted"])
		precision_2_weighted.append(elem[1]["precision_weighted"])

		recall_1_micro.append(elem[0]["recall_micro"])
		recall_2_micro.append(elem[1]["recall_micro"])

		recall_1_macro.append(elem[0]["recall_macro"])
		recall_2_macro.append(elem[1]["recall_macro"])

		recall_1_weighted.append(elem[0]["recall_weighted"])
		recall_2_weighted.append(elem[1]["recall_weighted"])

		mutual_information_1.append(elem[0]["mutual_info_score"])
		mutual_information_2.append(elem[1]["mutual_info_score"])

		normalized_mutual_information_1.append(elem[0]["normalized_mutual_info_score"])
		normalized_mutual_information_2.append(elem[1]["normalized_mutual_info_score"])

		adjusted_mutual_information_1.append(elem[0]["adjusted_mutual_info_score"])
		adjusted_mutual_information_2.append(elem[1]["adjusted_mutual_info_score"])

		homogeneity_1.append(elem[0]["homogeneity_score"])
		homogeneity_2.append(elem[1]["homogeneity_score"])

		silhoutte_level_1.append(elem[2]["level1"])
		dunn1_level_1.append(elem[3]["level1"])
		dunn2_level_1.append(elem[4]["level1"])
		dunn3_level_1.append(elem[5]["level1"])

		silhoutte_level_2 += elem[6]
		dunn1_level_2 += elem[7]
		dunn2_level_2 += elem[8]
		dunn3_level_2 += elem[9]

		viz = elem[10]
		for demonstration in viz.keys():
			utils.dict_insert_list(demonstration, viz[demonstration], list_of_frms)

	file = open(constants.PATH_TO_CLUSTERING_RESULTS + filename + ".txt", "wb")

	utils.print_and_write_2("precision_1_micro", np.mean(precision_1_micro), np.std(precision_1_micro), file)
	utils.print_and_write_2("precision_2_micro", np.mean(precision_2_micro), np.std(precision_2_micro), file)

	utils.print_and_write_2("precision_1_macro", np.mean(precision_1_macro), np.std(precision_1_macro), file)
	utils.print_and_write_2("precision_2_macro", np.mean(precision_2_macro), np.std(precision_2_macro), file)

	utils.print_and_write_2("precision_1_weighted", np.mean(precision_1_weighted), np.std(precision_1_weighted), file)
	utils.print_and_write_2("precision_2_weighted", np.mean(precision_2_weighted), np.std(precision_2_weighted), file)

	utils.print_and_write_2("recall_1_micro", np.mean(recall_1_micro), np.std(recall_1_micro), file)
	utils.print_and_write_2("recall_2_micro", np.mean(recall_2_micro), np.std(recall_2_micro), file)

	utils.print_and_write_2("recall_1_macro", np.mean(recall_1_macro), np.std(recall_1_macro), file)
	utils.print_and_write_2("recall_2_macro", np.mean(recall_2_macro), np.std(recall_2_macro), file)

	utils.print_and_write_2("recall_1_weighted", np.mean(recall_1_weighted), np.std(recall_1_weighted), file)
	utils.print_and_write_2("recall_2_weighted", np.mean(recall_2_weighted), np.std(recall_2_weighted), file)

	utils.print_and_write_2("mutual_info_1", np.mean(mutual_information_1), np.std(mutual_information_1), file)
	utils.print_and_write_2("mutual_info_2", np.mean(mutual_information_2), np.std(mutual_information_2), file)

	utils.print_and_write_2("normalized_mutual_info_1", np.mean(normalized_mutual_information_1), np.std(normalized_mutual_information_1), file)
	utils.print_and_write_2("normalized_mutual_info_2", np.mean(normalized_mutual_information_2), np.std(normalized_mutual_information_2), file)

	utils.print_and_write_2("adjusted_mutual_info_1", np.mean(adjusted_mutual_information_1), np.std(adjusted_mutual_information_1), file)
	utils.print_and_write_2("adjusted_mutual_info_2", np.mean(adjusted_mutual_information_2), np.std(adjusted_mutual_information_2), file)

	utils.print_and_write_2("homogeneity_1", np.mean(homogeneity_1), np.std(homogeneity_1), file)
	utils.print_and_write_2("homogeneity_2", np.mean(homogeneity_2), np.std(homogeneity_2), file)

	utils.print_and_write_2("silhoutte_level_1", np.mean(silhoutte_level_1), np.std(silhoutte_level_1), file)
	utils.print_and_write_2("silhoutte_level_2", np.mean(silhoutte_level_2), np.std(silhoutte_level_2), file)

	utils.print_and_write_2("dunn1_level1", np.mean(dunn1_level_1), np.std(dunn1_level_1), file)
	utils.print_and_write_2("dunn2_level1", np.mean(dunn2_level_1), np.std(dunn2_level_1), file)
	utils.print_and_write_2("dunn3_level1", np.mean(dunn3_level_1), np.std(dunn3_level_1), file)

	utils.print_and_write_2("dunn1_level2", np.mean(dunn1_level_2), np.std(dunn1_level_2), file)
	utils.print_and_write_2("dunn2_level2", np.mean(dunn2_level_2), np.std(dunn2_level_2), file)
	utils.print_and_write_2("dunn3_level2", np.mean(dunn3_level_2), np.std(dunn3_level_2), file)

	file.close()

	for demonstration in list_of_demonstrations:
		list_of_frms_demonstration = list_of_frms[demonstration]

		assert len(list_of_frms_demonstration) == len(list_of_demonstrations) - 1
		data = {}

		for i in range(len(list_of_frms_demonstration)):
			data[i] = list_of_frms_demonstration[0]

		broken_barh.plot_broken_barh(demonstration, data,
			constants.PATH_TO_CLUSTERING_RESULTS + demonstration +"_" + filename + ".jpg")


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	argparser.add_argument("feat_fname", help = "Pickle file of visual features", default = 4)
	argparser.add_argument("fname", help = "Pickle file of visual features", default = 4)
	args = argparser.parse_args()

	if args.debug == 'y':
		DEBUG = True
		list_of_demonstrations = ['Suturing_E001','Suturing_E002']
	else:
		DEBUG = False
		# list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
		# "Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

		# list_of_demonstrations = ["100_01", "100_02", "100_03", "100_04", "100_05"]

		# list_of_demonstrations = ["011_01", "011_02", "011_03", "011_04", "011_05"]

		# list_of_demonstrations = ["Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

		list_of_demonstrations = ["plane_3", "plane_4", "plane_5",
			"plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

		# list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']

		# list_of_demonstrations = ["0101_01", "0101_02", "0101_03", "0101_04", "0101_05"]


		# list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005']

		# list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
		# 'Suturing_D001','Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
		# 'Suturing_C001','Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
		# 'Suturing_F001','Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']

	combinations = get_list_of_demo_combinations(list_of_demonstrations)

	i = 1
	all_metrics = []

	# for _ in range(4):
	# 	print "---- k-Fold Cross Validation, Run "+ str(i) + " out of " + str(4) + " ----"
	# 	mc = MilestonesClustering(DEBUG, list_of_demonstrations, args.feat_fname, args.fname)
	# 	all_metrics.append(mc.do_everything())
	# 	i += 1

	for elem in combinations:	
		print "---- k-Fold Cross Validation, Run "+ str(i) + " out of " + str(len(combinations)) + " ----"
		mc = MilestonesClustering(DEBUG, list(elem), args.feat_fname, args.fname)
		all_metrics.append(mc.do_everything())
		i += 1
	print "----------- CALCULATING THE ODDS ------------"
	post_evaluation(all_metrics, args.fname, list_of_demonstrations, args.feat_fname)