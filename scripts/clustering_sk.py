#!/usr/bin/env python

import pickle
import numpy as np
import random
import os
import argparse
import sys
import IPython
import itertools

#Cluster Validity Indices
sys.path.insert(0, "/home/animesh/DeepMilestones/scripts/jqm_cvi")
import cvi

import constants_sk as constants
import parser
import utils
import cluster_evaluation

from sklearn import (mixture, preprocessing, neighbors, metrics, cross_decomposition)
from decimal import Decimal
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
mutual_info_score, homogeneity_score, completeness_score, recall_score, precision_score)
from sklearn.cross_decomposition import (CCA, PLSCanonical)

PATH_TO_FEATURES = constants.PATH_TO_DATA + constants.PROC_FEATURES_FOLDER

class MilestonesClustering():
	def __init__(self, DEBUG, list_of_demonstrations, featfile, trialname):
		# self.list_of_demonstrations = parser.generate_list_of_videos(constants.PATH_TO_DATA + constants.CONFIG_FILE)
	
		self.list_of_demonstrations = list_of_demonstrations
		self.data_X = {}
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
		self.map_cp_level1 = {}
		self.map_cp_level2 = {}
		self.map_level1_cp = {}
		self.map_level2_cp = {}
		self.map_cp2milestones = {}
		self.map_cp2surgemes = {}
		self.map_cp2surgemetransitions = {}
		self.l2_cluster_matrices = {}
		self.map_frm2surgeme = utils.get_all_frame2surgeme_maps(self.list_of_demonstrations)
		self.trial = utils.hashcode() + trialname
		# self.trial = trialname
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

		self.gmm_objects = {}

		self.sr = 10

	def construct_features(self):

		self.data_X = pickle.load(open(PATH_TO_FEATURES + str(self.featfile),"rb"))
		for demonstration in self.list_of_demonstrations:
			if demonstration not in self.data_X.keys():
				print "[ERROR] Missing demonstrations"
				sys.exit()

	def generate_transition_features(self):
		# print "Generating Transition Features"

		for demonstration in self.list_of_demonstrations:

			X = self.data_X[demonstration]
			self.data_X_size[demonstration] = X.shape[1]
			T = X.shape[0]
			N = utils.reshape(np.concatenate((X[0], X[1]), axis = 1))

			for t in range(T - 1):

				n_t = utils.reshape(np.concatenate((X[t], X[t + 1]), axis = 1))
				N = np.concatenate((N, n_t), axis = 0)

			self.data_N[demonstration] = N

	def generate_change_points(self):

		cp_index = 0

		for demonstration in self.list_of_demonstrations:

			# print "Changepoints for " + demonstration
			N = self.data_N[demonstration]
			# print N[0].shape

		for demonstration in self.list_of_demonstrations:

			# print "Changepoints for " + demonstration
			N = self.data_N[demonstration]

			gmm = mixture.GMM(n_components = 10, covariance_type='full')
			gmm.fit(N)
			Y = gmm.predict(N)
	
			self.save_cluster_metrics(N, Y, gmm.means_, 'cpts_' + demonstration, gmm)

			start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
				+ demonstration + "_capture2.p")

			size_of_X = self.data_X_size[demonstration]

			for i in range(len(Y) - 1):

				if Y[i] != Y[i + 1]:

					change_pt = N[i][size_of_X:]
					# print N.shape, change_pt.shape
					self.append_cp_array(change_pt)
					self.map_cp2frm[cp_index] = start + i * self.sr
					self.map_cp2demonstrations[cp_index] = demonstration
					self.list_of_cp.append(cp_index)

					cp_index += 1


	def append_cp_array(self, cp):
		if self.change_pts is None:
			self.change_pts = utils.reshape(cp)
			self.change_pts_W = utils.reshape(cp[:38])
			self.change_pts_Z = utils.reshape(cp[38:])

		else:
			try:
				self.change_pts = np.concatenate((self.change_pts, utils.reshape(cp)), axis = 0)
			except ValueError as e:
				print e
				sys.exit()
				# IPython.embed()
			self.change_pts_W = np.concatenate((self.change_pts_W, utils.reshape(cp[:38])), axis = 0)
			self.change_pts_Z = np.concatenate((self.change_pts_Z, utils.reshape(cp[38:])), axis = 0)


	def save_cluster_metrics(self, points, predictions, means, key, model, level2_mode = False):

		self.gmm_objects[key] = model

		# print key, level2_mode

		try:
			silhoutte = metrics.silhouette_score(points, predictions, metric='euclidean')
			self.silhouette_scores[key] = silhoutte
			if level2_mode:
				self.level2_silhoutte.append(silhoutte)

		except ValueError as e:
			pass

		dunn_scores = cluster_evaluation.dunn_index(points, predictions, means)

		self.dunn_scores_1[key] = dunn_scores[0]
		self.dunn_scores_2[key] = dunn_scores[1]
		self.dunn_scores_3[key] = dunn_scores[2]

		if level2_mode:
			self.level2_dunn_1.append(dunn_scores[0])
			self.level2_dunn_2.append(dunn_scores[1])
			self.level2_dunn_3.append(dunn_scores[2])

	def cluster_changepoints_level1(self):

		# print "Level1 : Clustering changepoints in Z(t)"

		gmm = mixture.GMM(n_components = 10, covariance_type='full')
		gmm.fit(self.change_pts_Z)

		Y = gmm.predict(self.change_pts_Z)

		self.save_cluster_metrics(self.change_pts_Z, Y, gmm.means_, 'level1', gmm)

		for i in range(len(Y)):
			label = constants.alphabet_map[Y[i] + 1]
			self.map_cp_level1[i] = label
			utils.dict_insert_list(label, i, self.map_level1_cp)

		self.generate_l2_cluster_matrices()

	def generate_l2_cluster_matrices(self):

		for key in sorted(self.map_level1_cp.keys()):

			list_of_cp = self.map_level1_cp[key]
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

		for key in sorted(self.map_level1_cp.keys()):
			mkdir_l1_cluster = mkdir_path + "/" + key

			list_of_cp_key = self.map_level1_cp[key]

			if self.check_pruning_condition(list_of_cp_key):
				continue

			os.mkdir(mkdir_l1_cluster)

		for key in sorted(self.map_level1_cp.keys()):
			matrix = self.l2_cluster_matrices[key]
			list_of_cp_key = self.map_level1_cp[key]

			if self.check_pruning_condition(list_of_cp_key):
				self.pruned_L1_clusters.append(key)
				del self.map_level1_cp[key]
				for pruned_cp in list_of_cp_key:
					self.list_of_cp.remove(pruned_cp)
				continue

			n_components = 3
			gmm = mixture.GMM(n_components = n_components, covariance_type='full')


			try:
				gmm.fit(matrix)
				self.gmm_objects[key] = gmm
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
					# IPython.embed()

				self.map_cp2milestones[cp] = milestone

				self.file.write("%s             %3d         %s   %3d   %3d    %3d\n" % (l1_cluster, l2_cluster, demonstration, frm, cp, surgeme))

				self.copy_frames(demonstration, frm, str(l1_cluster), str(l2_cluster), surgeme)

			self.copy_milestone_frames(matrix, list_of_cp_key, gmm)

	def copy_milestone_frames(self, matrix, list_of_cp_key, gmm):
		neigh = neighbors.KNeighborsClassifier(n_neighbors = 1)
		neigh.fit(matrix, list_of_cp_key)
		milestone_closest_cp = neigh.predict(gmm.means_)

		assert len(milestone_closest_cp) == 3

		for cp in milestone_closest_cp:
			demonstration = self.map_cp2demonstrations[cp]
			surgeme = self.map_frm2surgeme[demonstration][self.map_cp2frm[cp]]
			frm = utils.get_frame_fig_name(self.map_cp2frm[cp])

			from_path = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_capture2/" + frm

			to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/milestones/" + self.map_cp2milestones[cp] + "_" + str(surgeme) + "_" + demonstration + "_" + frm

			utils.sys_copy(from_path, to_path)

	# Prune clusters which represent less than 20% of the total demonstration data
	def check_pruning_condition(self, list_of_cp_key):
		demonstrations_in_cluster = [self.map_cp2demonstrations[cp] for cp in list_of_cp_key]

		num_demonstrations = len(set(demonstrations_in_cluster))
		num_total_demonstrations = len(self.list_of_demonstrations)
		data_representation = num_demonstrations / float(num_total_demonstrations)

		print data_representation, len(list_of_cp_key)

		return data_representation <= 0.3

	def copy_frames(self, demonstration, frm, l1_cluster, l2_cluster, surgeme):

		from_path = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_capture2/" + utils.get_frame_fig_name(frm)

		to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/" + l1_cluster + "/" + l2_cluster + "_" + str(surgeme) + "_" + demonstration + "_" + utils.get_frame_fig_name(frm)

		utils.sys_copy(from_path, to_path)

	def cluster_evaluation(self):

		for cp in self.list_of_cp:

			demonstration = self.map_cp2demonstrations[cp]
			frm = self.map_cp2frm[cp]

			curr_surgeme = self.map_frm2surgeme[demonstration][frm]
			self.map_cp2surgemes[cp] = curr_surgeme

			ranges = sorted(utils.get_annotation_segments(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
				+ demonstration + "_capture2.p"))

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
				surgemetransition = str(curr_surgeme) + "->" + str(prev_surgeme)

			self.map_cp2surgemetransitions[cp] = surgemetransition

		self.cp_surgemes = set(self.map_cp2surgemes.values())


		# Initialize data structures
		table = {}
		for L1_cluster in self.map_level1_cp.keys():
			new_dict = {}
			for surgeme in self.cp_surgemes:
				new_dict[surgeme] = 0
			table[L1_cluster] = new_dict

		surgeme_count = {}
		for surgeme in self.cp_surgemes:
			surgeme_count[surgeme] = 0

		for L1_cluster in self.map_level1_cp.keys():
			list_of_cp_key = self.map_level1_cp[L1_cluster]
			for cp in list_of_cp_key:
				surgeme = self.map_frm2surgeme[self.map_cp2demonstrations[cp]][self.map_cp2frm[cp]]
				surgeme_count[surgeme] += 1

				curr_dict = table[L1_cluster]
				curr_dict[surgeme] += 1
				table[L1_cluster] = curr_dict

		final_clusters = list(set(self.map_level1_cp.keys()) - set(self.pruned_L1_clusters))

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
		labels_pred_1 = []
		labels_pred_2 = []
		labels_true = []

		for cp in self.list_of_cp:
			labels_true.append(self.map_cp2surgemetransitions[cp])

			milestone_label = self.map_cp2milestones[cp]
			labels_pred_1.append(milestone_label)
			labels_pred_2.append(list(milestone_label)[0])

		assert len(labels_true) == len(labels_pred_1) == len(labels_pred_2)

		return labels_true, labels_pred_1, labels_pred_2

	def cluster_metrics(self):
		labels_true, labels_pred_1, labels_pred_2 = self.prepare_labels()

		metric_funcs = [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, 
		mutual_info_score, homogeneity_score, completeness_score]

		# ------ Label-based metrics ------
		# print("\n\nPred = L2 Milestones       Pred= L1 Labels      Metric\n\n")
		self.file.write("\n\nPred = L2 Milestones       Pred= L1 Labels     Metric\n\n")

		for metric in metric_funcs:

			try:
				score_1 = round(Decimal(metric(labels_true, labels_pred_1)), 2)
				score_2 = round(Decimal(metric(labels_true, labels_pred_2)), 2)

			except ValueError as e:
				print e
				print "Pruning error"
				sys.exit()
			key =  repr(metric).split()[1]
			self.label_based_scores_1[key] = score_1
			self.label_based_scores_2[key] = score_2

			self.file.write("%3.3f        %3.3f        %s\n" % (score_1, score_2, key))
			# print("%3.3f        %3.3f        %s\n" % (score_1, score_2, key))

		self.file.write("\nSilhouette Scores\n")
		# print("\nSilhoutte scores\n")

		# ------ Silhouette Scores ------
		for layer in sorted(self.silhouette_scores):
			score = self.silhouette_scores[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))
			# print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		self.file.write("\nDunn Scores1\n")
		# print("\nDunn scores1\n")

		# ------ Dunn Scores ------
		for layer in sorted(self.dunn_scores_1):
			score = self.dunn_scores_1[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))
			# print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		self.file.write("\nDunn Scores2\n")
		# print("\nDunn scores2\n")

		# ------ Dunn Scores ------
		for layer in sorted(self.dunn_scores_2):
			score = self.dunn_scores_2[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))
			# print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		self.file.write("\nDunn Scores3\n")
		# print("\nDunn scores3\n")

		# ------ Dunn Scores ------
		for layer in sorted(self.dunn_scores_3):
			score = self.dunn_scores_3[layer]
			self.file.write("%3.3f        %s\n" % (round(Decimal(score), 2), layer))
			# print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		data = [self.label_based_scores_1, self.label_based_scores_2,
		self.silhouette_scores, self.dunn_scores_1, self.dunn_scores_2, self.dunn_scores_3,
		self.level2_silhoutte, self.level2_dunn_1, self.level2_dunn_2, self.level2_dunn_3]

		pickle.dump(data, open(self.metrics_picklefile, "wb"))

		return data

	def clean_up(self):
		self.file.close()

	def do_everything(self):

		self.construct_features()

		self.generate_transition_features()

		self.generate_change_points()

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

def parse_metrics(metrics, filename):

	mutual_information_1 = []
	homogeneity_1 = []
	mutual_information_2 = []
	homogeneity_2 = []

	silhoutte_level_1 = []
	dunn1_level_1 = []
	dunn2_level_1 = []
	dunn3_level_1 = []

	silhoutte_level_2 = []
	dunn1_level_2 = []
	dunn2_level_2 = []
	dunn3_level_2 = []

	for elem in metrics:
		mutual_information_1.append(elem[0]["mutual_info_score"])
		mutual_information_2.append(elem[1]["mutual_info_score"])
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

	file = open(constants.PATH_TO_CLUSTERING_RESULTS + filename + ".txt", "wb")

	utils.print_and_write_2("mutual_info_1", np.mean(mutual_information_1), np.std(mutual_information_1), file)
	utils.print_and_write_2("mutual_info_2", np.mean(mutual_information_2), np.std(mutual_information_2), file)
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

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	argparser.add_argument("feat_fname", help = "Pickle file of visual features", default = 4)
	argparser.add_argument("metric_fname", help = "Pickle file of visual features", default = 4)
	args = argparser.parse_args()

	if args.debug == 'y':
		DEBUG = True
		list_of_demonstrations = ['Suturing_E001','Suturing_E002']
	else:
		DEBUG = False
		# list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']
		list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
		'Suturing_D001','Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
		'Suturing_C001','Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
		'Suturing_F001','Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']

	combinations = get_list_of_demo_combinations(list_of_demonstrations)

	i = 1
	all_metrics = []
	for elem in combinations:	
		print "---- "+ str(i) + " ----"
		mc = MilestonesClustering(DEBUG, list(elem), args.feat_fname, args.metric_fname)
		all_metrics.append(mc.do_everything())
		i += 1
	# print "----------- CALCULATING THE ODDS ------------"
	parse_metrics(all_metrics, args.metric_fname)
