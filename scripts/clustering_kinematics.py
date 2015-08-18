#!/usr/bin/env python

import pickle
import numpy as np
import argparse
import IPython
import itertools
import sys
from decimal import Decimal

import constants
import parser
import utils
import cluster_evaluation
import featurization

from sklearn import (mixture, neighbors, metrics)
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
mutual_info_score, homogeneity_score, completeness_score, recall_score, precision_score)

PATH_TO_FEATURES = constants.PATH_TO_DATA + constants.PROC_FEATURES_FOLDER

class KinematicsClustering():
	def __init__(self, DEBUG, list_of_demonstrations, fname, log, vision_mode = False, feat_fname = None):
		self.list_of_demonstrations = list_of_demonstrations

		if vision_mode and feat_fname is None:
			print "[Error] Please provide file with visual features"
			sys.exit()

		self.vision_mode = vision_mode
		self.feat_fname = feat_fname
		self.data_X = {}
		self.X_dimension = 0
		self.data_X_size = {}
		self.data_N = {}
		self.log = log

		self.file = None
		self.metrics_picklefile = constants.PATH_TO_CLUSTERING_RESULTS + fname + ".p"

		self.changepoints = None

		self.list_of_cp = []
		self.map_cp2frm = {}
		self.map_cp2demonstrations = {}
		self.map_cp2cluster = {}
		self.map_level1_cp = {}
		self.map_cp2milestones = {}
		self.map_cp2surgemes = {}
		self.map_cp2surgemetransitions = {}
		self.map_frm2surgeme = parser.get_all_frame2surgeme_maps(self.list_of_demonstrations)
		self.trial = utils.hashcode() + fname
		# self.trial = fname
		self.cp_surgemes = []
		self.pruned_L1_clusters = []

		self.silhouette_scores = {}
		self.dunn_scores_1 = {}
		self.dunn_scores_2 = {}
		self.dunn_scores_3 = {}

		self.label_based_scores_1 = {}

		self.gmm_objects = {}

		self.sr = constants.SR

		# utils.print_and_write("Dumping metrics to file: " + self.metrics_picklefile, self.log)

	def construct_features_visual(self):

		self.data_X = pickle.load(open(PATH_TO_FEATURES + str(self.feat_fname),"rb"))
		for demonstration in self.list_of_demonstrations:
			if demonstration not in self.data_X.keys():
				print "[ERROR] Missing demonstrations"
				sys.exit()
			X = self.data_X[demonstration]
			X_visual = None
			for i in range(len(X)):
				X_visual = utils.safe_concatenate(X_visual, utils.reshape(X[i][38:]))
			assert X_visual.shape[0] == X.shape[0]

			self.data_X[demonstration] = X_visual

	def construct_features_kinematics(self):

		for demonstration in self.list_of_demonstrations:
			self.data_X[demonstration] = utils.sample_matrix(featurization.get_kinematic_features(demonstration), sampling_rate = self.sr)

	def generate_transition_features(self):
		print "Generating Transition Features"

		self.X_dimension = self.data_X[self.list_of_demonstrations[0]].shape[1]
		print "X dimension", str(self.X_dimension)

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

			print "Changepoints for " + demonstration
			N = self.data_N[demonstration]

			gmm = mixture.GMM(n_components = 10, covariance_type='full')
			gmm.fit(N)
			Y = gmm.predict(N)

			start, end = parser.get_start_end_annotations(constants.PATH_TO_DATA +
				constants.ANNOTATIONS_FOLDER + demonstration + "_capture2.p")
	
			self.save_cluster_metrics(N, Y, gmm.means_, 'cpts_' + demonstration, gmm)

			for i in range(len(Y) - 1):

				if Y[i] != Y[i + 1]:

					change_pt = N[i][self.X_dimension:]
					self.append_cp_array(utils.reshape(change_pt))
					self.map_cp2frm[cp_index] = start + i * self.sr
					self.map_cp2demonstrations[cp_index] = demonstration
					self.list_of_cp.append(cp_index)

					cp_index += 1

	def append_cp_array(self, cp):
		self.changepoints = utils.safe_concatenate(self.changepoints, cp)

	def save_cluster_metrics(self, points, predictions, means, key, model):

		self.gmm_objects[key] = model
		silhoutte = metrics.silhouette_score(points, predictions, metric='euclidean')
		self.silhouette_scores[key] = silhoutte

		dunn_scores = cluster_evaluation.dunn_index(points, predictions, means)

		self.dunn_scores_1[key] = dunn_scores[0]
		self.dunn_scores_2[key] = dunn_scores[1]
		self.dunn_scores_3[key] = dunn_scores[2]

	def cluster_changepoints(self):

		print "Clustering changepoints..."

		gmm = mixture.GMM(n_components = 10, covariance_type='full')
		gmm.fit(self.changepoints)

		predictions = gmm.predict(self.changepoints)

		self.save_cluster_metrics(self.changepoints, predictions, gmm.means_, 'level1', gmm)

		for i in range(len(predictions)):
			label = constants.alphabet_map[predictions[i] + 1]
			self.map_cp2cluster[i] = label
			utils.dict_insert_list(label, i, self.map_level1_cp)
			demonstration = self.map_cp2demonstrations[i]
			frm = self.map_cp2frm[i]
			try:
				surgeme = self.map_frm2surgeme[demonstration][frm]
			except KeyError as e:
				print e
				sys.exit()

			utils.print_and_write(("%3d   %s   %s   %3d   %3d\n" % (i, label, demonstration, frm, surgeme)), self.log)

	def check_pruning_condition(self, list_of_cp_key):
		"""
		Prune clusters which represent less than 20 percent of the total demonstration data
		"""

		demonstrations_in_cluster = [self.map_cp2demonstrations[cp] for cp in list_of_cp_key]

		num_demonstrations = len(set(demonstrations_in_cluster))
		num_total_demonstrations = len(self.list_of_demonstrations)
		data_representation = num_demonstrations / float(num_total_demonstrations)

		print data_representation

		return data_representation < 0.8

	def cluster_evaluation(self):

		for cp in self.list_of_cp:
			demonstration = self.map_cp2demonstrations[cp]
			frm = self.map_cp2frm[cp]

			curr_surgeme = self.map_frm2surgeme[demonstration][frm]
			self.map_cp2surgemes[cp] = curr_surgeme

			ranges = sorted(parser.get_annotation_segments(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
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

		utils.print_and_write('\n\n ---Confusion Matrix--- \n\n', self.log)
		utils.print_and_write(confusion_matrix, self.log)

		confusion_matrix = ""
		for L1_cluster in final_clusters:
			confusion_matrix = confusion_matrix + "\n" + L1_cluster + "   "
			for surgeme in self.cp_surgemes:
				# confusion_matrix += str(float("{0:.2f}".format(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])))) + "   "
				confusion_matrix += str(round(Decimal(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])), 2)) + "   "
			confusion_matrix += '\n'

		utils.print_and_write(confusion_matrix, self.log)
		utils.print_and_write("\n\n ---Surgeme Count--- \n\n", self.log)
		utils.print_and_write(repr(surgeme_count), self.log)
		utils.print_and_write("\n\n", self.log)

	def prepare_labels(self):
		labels_pred = []
		labels_true = []

		for cp in self.list_of_cp:
			labels_true.append(self.map_cp2surgemetransitions[cp])
			labels_pred.append(self.map_cp2cluster[cp])

		assert len(labels_true) == len(labels_pred)

		return labels_true, labels_pred

	def cluster_metrics(self):
		labels_true, labels_pred = self.prepare_labels()

		metric_funcs = [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, 
		mutual_info_score, homogeneity_score, completeness_score]

		# ------ Label-based metrics ------
		utils.print_and_write("\n\nPred= L1 Labels     Metric\n\n", self.log)

		for metric in metric_funcs:

			score_1 = round(Decimal(metric(labels_true, labels_pred)), 2)
			key =  repr(metric).split()[1]
			self.label_based_scores_1[key] = score_1

			utils.print_and_write(("%3.3f        %s\n" % (score_1, key)), self.log)

		utils.print_and_write("\nSilhoutte scores\n", self.log)

		# ------ Silhouette Scores ------
		for layer in sorted(self.silhouette_scores):
			score = self.silhouette_scores[layer]
			utils.print_and_write("%3.3f        %s\n" % (round(Decimal(score), 2), layer), self.log)

		utils.print_and_write("\nDunn Scores1\n", self.log)

		# ------ Dunn Scores # 1------
		for layer in sorted(self.dunn_scores_1):
			score = self.dunn_scores_1[layer]
			utils.print_and_write("%3.3f        %s\n" % (round(Decimal(score), 2), layer), self.log)

		utils.print_and_write("\nDunn Scores2\n", self.log)

		# ------ Dunn Scores # 2------
		for layer in sorted(self.dunn_scores_2):
			score = self.dunn_scores_2[layer]
			utils.print_and_write("%3.3f        %s\n" % (round(Decimal(score), 2), layer), self.log)

		utils.print_and_write("\nDunn Scores3\n", self.log)

		# ------ Dunn Scores #3 ------
		for layer in sorted(self.dunn_scores_3):
			score = self.dunn_scores_3[layer]
			utils.print_and_write("%3.3f        %s\n" % (round(Decimal(score), 2), layer), self.log)

		data = [self.label_based_scores_1, self.silhouette_scores, self.dunn_scores_1,
		self.dunn_scores_2, self.dunn_scores_3]

		# pickle.dump(data, open(self.metrics_picklefile, "wb"))

		return data

	def do_everything(self):

		if self.vision_mode:
			self.construct_features_visual()
		else:
			self.construct_features_kinematics()

		self.generate_transition_features()

		self.generate_change_points()

		self.cluster_changepoints()

		self.cluster_evaluation()

		data = self.cluster_metrics()

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

def parse_metrics(metrics, file):

	mutual_information_1 = []
	normalized_mutual_information_1 = []
	adjusted_mutual_information_1 = []
	homogeneity_1 = []

	silhoutte_level_1 = []
	dunn1_level_1 = []
	dunn2_level_1 = []
	dunn3_level_1 = []

	for elem in metrics:

		mutual_information_1.append(elem[0]["mutual_info_score"])
		normalized_mutual_information_1.append(elem[0]["normalized_mutual_info_score"])
		adjusted_mutual_information_1.append(elem[0]["adjusted_mutual_info_score"])
		homogeneity_1.append(elem[0]["homogeneity_score"])

		silhoutte_level_1.append(elem[1]["level1"])
		dunn1_level_1.append(elem[2]["level1"])
		dunn2_level_1.append(elem[3]["level1"])
		dunn3_level_1.append(elem[4]["level1"])

	utils.print_and_write_2("mutual_info", np.mean(mutual_information_1), np.std(mutual_information_1), file)
	utils.print_and_write_2("normalized_mutual_info", np.mean(normalized_mutual_information_1), np.std(normalized_mutual_information_1), file)
	utils.print_and_write_2("adjusted_mutual_info", np.mean(adjusted_mutual_information_1), np.std(adjusted_mutual_information_1), file)

	utils.print_and_write_2("homogeneity", np.mean(homogeneity_1), np.std(homogeneity_1), file)
	utils.print_and_write_2("silhoutte_level_1", np.mean(silhoutte_level_1), np.std(silhoutte_level_1), file)

	utils.print_and_write_2("dunn1", np.mean(dunn1_level_1), np.std(dunn1_level_1), file)
	utils.print_and_write_2("dunn2", np.mean(dunn2_level_1), np.std(dunn2_level_1), file)
	utils.print_and_write_2("dunn3", np.mean(dunn3_level_1), np.std(dunn3_level_1), file)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	argparser.add_argument("--visual", help = "Debug mode?[y/n]", default = False)
	argparser.add_argument("fname", help = "Pickle file of visual features", default = 4)
	args = argparser.parse_args()

	if args.debug == 'y':
		DEBUG = True
		list_of_demonstrations = ['Suturing_E001','Suturing_E002']
	else:
		DEBUG = False
		list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']
		# list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
		# 'Suturing_D001','Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
		# 'Suturing_C001','Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
		# 'Suturing_F001','Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']


	vision_mode = False
	feat_fname = None
	if args.visual:
		vision_mode = True
		feat_fname = args.visual

	combinations = get_list_of_demo_combinations(list_of_demonstrations)

	all_metrics = []
	log = open(constants.PATH_TO_CLUSTERING_RESULTS + args.fname + ".txt", "wb")

	for i in range(len(combinations)):
		utils.print_and_write("\n----------- Combination #" + str(i) + " -------------\n", log)

		mc = KinematicsClustering(DEBUG, list(combinations[i]), args.fname + str(i), log, vision_mode, feat_fname)
		all_metrics.append(mc.do_everything())

	print "----------- CALCULATING THE ODDS ------------"
	parse_metrics(all_metrics, log)

	log.close()