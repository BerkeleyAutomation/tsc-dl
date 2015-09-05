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
import broken_barh
import time

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
		self.data_X_size = 0
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

		self.silhouette_score = None
		self.dunn_scores_1 = {}
		self.dunn_scores_2 = {}
		self.dunn_scores_3 = {}

		self.label_based_scores_1 = {}

		self.sr = constants.SR

		# Components for Mixture model at each level
		if self.vision_mode:
			self.n_components_cp = constants.N_COMPONENTS_CP_Z
			self.n_components_L1 = constants.N_COMPONENTS_L1_Z
			self.temporal_window = constants.TEMPORAL_WINDOW_Z
			self.representativeness = constants.PRUNING_FACTOR_Z
			self.ALPHA_CP = constants.ALPHA_Z_CP

		else:
			self.n_components_cp = constants.N_COMPONENTS_CP_W
			self.n_components_L1 = constants.N_COMPONENTS_L1_W
			self.temporal_window = constants.TEMPORAL_WINDOW_W
			self.representativeness = constants.PRUNING_FACTOR_W
			self.ALPHA_CP = constants.ALPHA_W_CP

	def construct_features_visual(self):

		data_X = pickle.load(open(PATH_TO_FEATURES + str(self.feat_fname),"rb"))
		for demonstration in self.list_of_demonstrations:
			if demonstration not in data_X.keys():
				print "[ERROR] Missing demonstrations"
				sys.exit()
			X = data_X[demonstration]
			X_visual = None
			for i in range(len(X)):
				X_visual = utils.safe_concatenate(X_visual, utils.reshape(X[i][constants.KINEMATICS_DIM:]))
			assert X_visual.shape[0] == X.shape[0]

			self.data_X[demonstration] = X_visual

	def construct_features_kinematics(self):

		for demonstration in self.list_of_demonstrations:
			self.data_X[demonstration] = utils.sample_matrix(parser.get_kinematic_features(demonstration), sampling_rate = self.sr)
			print self.data_X[demonstration].shape

	def generate_transition_features(self):

		self.X_dimension = self.data_X[self.list_of_demonstrations[0]].shape[1]
		print "X dimension", str(self.X_dimension)

		for demonstration in self.list_of_demonstrations:

			X = self.data_X[demonstration]
			T = X.shape[0]
			N = None

			for t in range(T - self.temporal_window):
				n_t = utils.make_transition_feature(X, self.temporal_window, t)
				N = utils.safe_concatenate(N, n_t)

			self.data_N[demonstration] = N

	def generate_change_points_1(self):
		"""
		Generates changespoints by clustering within demonstration.
		"""
		cp_index = 0

		for demonstration in self.list_of_demonstrations:

			print "Changepoints for " + demonstration
			N = self.data_N[demonstration]

			gmm = mixture.GMM(n_components = self.n_components_cp, n_iter=5000, thresh = 5e-5, covariance_type='full')
			gmm.fit(N)
			Y = gmm.predict(N)

			start, end = parser.get_start_end_annotations(constants.PATH_TO_DATA +
				constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p")
	
			self.save_cluster_metrics(N, Y, gmm.means_, 'cpts_' + demonstration, gmm)

			for i in range(len(Y) - 1):

				if Y[i] != Y[i + 1]:

					change_pt = N[i][self.X_dimension:]
					self.append_cp_array(utils.reshape(change_pt))
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

		print "Generating Changepoints. Fitting GMM ..."

		if constants.REMOTE == 1:
			print "Init DPGMM"
			#DO NOT FIDDLE WITH PARAMS WITHOUT CONSENT
			avg_len = int(big_N.shape[0]/len(self.list_of_demonstrations))
			DP_GMM_COMPONENTS = int(avg_len/25) #tuned with suturing experts only for kinematics
			print DP_GMM_COMPONENTS, "ALPHA: ", self.ALPHA_CP
			dpgmm = mixture.DPGMM(n_components = DP_GMM_COMPONENTS, covariance_type='diag', n_iter = 100, alpha = self.ALPHA_CP, thresh= 1e-7)

			# avg_len = int(big_N.shape[0]/len(self.list_of_demonstrations))
			# DP_GMM_COMPONENTS =int(avg_len/25) #tuned with suturing experts only for video			
			# print DP_GMM_COMPONENTS			
			# dpgmm = mixture.DPGMM(n_components = DP_GMM_COMPONENTS, covariance_type='diag', n_iter = 100, alpha = 1e-3, thresh= 1e-7)


			print "Init GMM"
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', n_iter=5000, thresh = 5e-5)

		if constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', tol = 0.01)

		else:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full')

		print "Fitting GMM"
		start = time.time()
		gmm.fit(big_N)
		end = time.time()
		print "GMM Time:", end - start
		print "Fitting DPGMM"
		start = time.time()
		dpgmm.fit(big_N)
		end = time.time()
		print "DPGMM Time:", end - start

		Y_gmm = gmm.predict(big_N)
		Y_dpgmm = dpgmm.predict(big_N)

		# IPython.embed()
		Y = Y_dpgmm

		print "Num clusters in Changepoint clusters, DPGMM: ", len(set(Y)), " GMM: ", len(set(Y_gmm))
		print "Done fitting GMM..."

		for w in range(len(Y) - 1):

			if Y[w] != Y[w + 1]:
				change_pt = big_N[w][self.X_dimension:]
				self.append_cp_array(utils.reshape(change_pt))
				self.map_cp2frm[cp_index] = map_index2frm[w]
				self.map_cp2demonstrations[cp_index] = map_index2demonstration[w]
				self.list_of_cp.append(cp_index)

				cp_index += 1

		print "Done with generating change points, " + str(cp_index)

	def append_cp_array(self, cp):
		self.changepoints = utils.safe_concatenate(self.changepoints, cp)

	def save_cluster_metrics(self, points, predictions, means, key, model):

		if key == 'level1':
			self.silhouette_score = metrics.silhouette_score(points, predictions, metric='euclidean')

		# dunn_scores = cluster_evaluation.dunn_index(points, predictions, means)

		dunn_scores = [0,0,0]

		if (dunn_scores[0] is not None) and (dunn_scores[1] is not None) and (dunn_scores[2] is not None):

			self.dunn_scores_1[key] = dunn_scores[0]
			self.dunn_scores_2[key] = dunn_scores[1]
			self.dunn_scores_3[key] = dunn_scores[2]

	def cluster_changepoints(self):

		print "Clustering changepoints..."

		if constants.REMOTE == 1:
			dpgmm = mixture.DPGMM(n_components = int(len(self.list_of_cp)/3), covariance_type='diag', n_iter = 10000, alpha = 0.4, thresh= 1e-4)
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', n_iter=5000, thresh = 0.01)
		elif constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', tol = 0.01)
		else:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full')
		
		gmm.fit(self.changepoints)
		predictions_gmm = gmm.predict(self.changepoints)
		
		predictions = []
		while True:
			print "Inside loop"
			dpgmm.fit(self.changepoints)
			predictions = dpgmm.predict(self.changepoints)
			# IPython.embed()
			if len(set(predictions))>1:
				break

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

	def cluster_pruning(self):
		for cluster in self.map_level1_cp.keys():
			cluster_list_of_cp = self.map_level1_cp[cluster]
			cluster_demonstrations = []
			for cp in cluster_list_of_cp:
				cluster_demonstrations.append(self.map_cp2demonstrations[cp])
			data_representation = float(len(set(cluster_demonstrations))) / float(len(self.list_of_demonstrations))
			print str(cluster) + ": " + str(data_representation), " " + str(len(cluster_list_of_cp))
			if data_representation <= self.representativeness:
				new_cluster_list = cluster_list_of_cp[:]
				for cp in cluster_list_of_cp:
					print "Pruning " + str(cluster) + " " + str(data_representation) +  ": " + str(cp)
					self.list_of_cp.remove(cp)
					new_cluster_list.remove(cp)
				self.map_level1_cp[cluster] = new_cluster_list

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
		labels_pred_ = []
		labels_true_ = []

		for cp in self.list_of_cp:
			labels_true_.append(self.map_cp2surgemetransitions[cp])
			labels_pred_.append(self.map_cp2cluster[cp])

		labels_pred = utils.label_convert_to_numbers(labels_pred_)
		labels_true = utils.label_convert_to_numbers(labels_true_)

		assert len(labels_true) == len(labels_pred)
		assert len(labels_true_) == len(labels_pred_)

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

		# ------ Precision & Recall ------
		utils.print_and_write("\n Precision & Recall scores \n", self.log)
		for ave in ["micro", "macro", "weighted"]:
			key = "precision_" + ave
			score_1 = utils.nsf(precision_score(labels_true, labels_pred, average = ave))

			self.label_based_scores_1[key] = score_1

			utils.print_and_write("%3.3f        %s\n" % (round(Decimal(score_1), 2), key), self.log)

			key = "recall_" + ave
			score_1 = utils.nsf(recall_score(labels_true, labels_pred, average = ave))

			self.label_based_scores_1[key] = score_1

			utils.print_and_write("%3.3f        %s\n" % (round(Decimal(score_1), 2), key), self.log)

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

		# ------ Visualizing changepoints on broken barh ------
		viz = {}

		for cp in self.list_of_cp:
			utils.dict_insert_list(self.map_cp2demonstrations[cp], self.map_cp2frm[cp], viz)

		data = [self.label_based_scores_1, self.silhouette_score, self.dunn_scores_1,
		self.dunn_scores_2, self.dunn_scores_3, viz]

		# pickle.dump(data, open(self.metrics_picklefile, "wb"))

		return data

	def do_everything(self):

		if self.vision_mode:
			self.construct_features_visual()
		else:
			self.construct_features_kinematics()

		self.generate_transition_features()

		self.generate_change_points_2() #cluster over the full data set
		# self.generate_change_points_1() #cluster over the each demo

		self.cluster_changepoints()

		self.cluster_pruning()

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

def post_evaluation(metrics, file, fname, vision_mode):

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
	silhouette_scores = []

	dunn1_level_1 = []
	dunn2_level_1 = []
	dunn3_level_1 = []

	list_of_frms = {}

	for elem in metrics:

		mutual_information_1.append(elem[0]["mutual_info_score"])
		normalized_mutual_information_1.append(elem[0]["normalized_mutual_info_score"])
		adjusted_mutual_information_1.append(elem[0]["adjusted_mutual_info_score"])
		homogeneity_1.append(elem[0]["homogeneity_score"])

		silhouette_scores.append(elem[1])

		dunn1_level_1.append(elem[2]["level1"])
		dunn2_level_1.append(elem[3]["level1"])
		dunn3_level_1.append(elem[4]["level1"])

		precision_1_micro.append(elem[0]["precision_micro"])
		precision_1_macro.append(elem[0]["precision_macro"])
		precision_1_weighted.append(elem[0]["precision_weighted"])

		recall_1_micro.append(elem[0]["recall_micro"])
		recall_1_macro.append(elem[0]["recall_macro"])
		recall_1_weighted.append(elem[0]["recall_weighted"])

		viz = elem[5]
		for demonstration in viz.keys():
			utils.dict_insert_list(demonstration, viz[demonstration], list_of_frms)

	utils.print_and_write_2("precision_1_micro", np.mean(precision_1_micro), np.std(precision_1_micro), file)
	utils.print_and_write_2("precision_1_macro", np.mean(precision_1_macro), np.std(precision_1_macro), file)
	utils.print_and_write_2("precision_1_weighted", np.mean(precision_1_weighted), np.std(precision_1_weighted), file)

	utils.print_and_write_2("recall_1_micro", np.mean(recall_1_micro), np.std(recall_1_micro), file)
	utils.print_and_write_2("recall_1_macro", np.mean(recall_1_macro), np.std(recall_1_macro), file)
	utils.print_and_write_2("recall_1_weighted", np.mean(recall_1_weighted), np.std(recall_1_weighted), file)

	utils.print_and_write_2("mutual_info", np.mean(mutual_information_1), np.std(mutual_information_1), file)
	utils.print_and_write_2("normalized_mutual_info", np.mean(normalized_mutual_information_1), np.std(normalized_mutual_information_1), file)
	utils.print_and_write_2("adjusted_mutual_info", np.mean(adjusted_mutual_information_1), np.std(adjusted_mutual_information_1), file)
	utils.print_and_write_2("silhouette_scores", np.mean(silhouette_scores), np.std(silhouette_scores), file)

	utils.print_and_write_2("homogeneity", np.mean(homogeneity_1), np.std(homogeneity_1), file)

	utils.print_and_write_2("dunn1", np.mean(dunn1_level_1), np.std(dunn1_level_1), file)
	utils.print_and_write_2("dunn2", np.mean(dunn2_level_1), np.std(dunn2_level_1), file)
	utils.print_and_write_2("dunn3", np.mean(dunn3_level_1), np.std(dunn3_level_1), file)

	list_of_dtw_values = []
	list_of_norm_dtw_values = []
	list_of_lengths = []

	if vision_mode:
		T = constants.N_COMPONENTS_TIME_Z
	else:
		T = constants.N_COMPONENTS_TIME_W

	for demonstration in list_of_demonstrations:
		list_of_frms_demonstration = list_of_frms[demonstration]

		assert len(list_of_frms_demonstration) == len(list_of_demonstrations) - 1
		data = {}

		for i in range(len(list_of_frms_demonstration)):
			data[i] = list_of_frms_demonstration[0]

		dtw_score, normalized_dtw_score, length = broken_barh.plot_broken_barh(demonstration, data,
			constants.PATH_TO_CLUSTERING_RESULTS + demonstration +"_" + fname + ".jpg", T)
		list_of_dtw_values.append(dtw_score)
		list_of_norm_dtw_values.append(normalized_dtw_score)
		list_of_lengths.append(length)

	utils.print_and_write_2("dtw_score", np.mean(list_of_dtw_values), np.std(list_of_dtw_values), file)
	utils.print_and_write_2("dtw_score_normalized", np.mean(list_of_norm_dtw_values), np.std(list_of_norm_dtw_values), file)
	utils.print_and_write(str(list_of_lengths), file)

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

		# list_of_demonstrations = ["010_01", "010_02", "010_03", "010_04", "010_05"]

		# list_of_demonstrations = ["100_01", "100_02", "100_03", "100_04", "100_05"]

		# list_of_demonstrations = ["baseline_000", "baseline_010", "baseline_025", "baseline_050", "baseline_075"]

		# list_of_demonstrations = ["baseline2_010_01", "baseline2_010_02", "baseline2_010_03", "baseline2_010_04", "baseline2_010_05"]

		# list_of_demonstrations = ["Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

		# list_of_demonstrations = ["plane_3", "plane_4", "plane_5",
		# 	"plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

		# list_of_demonstrations = ["plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

		# list_of_demonstrations = ["plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

		# list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
		# "Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

		list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']


		# list_of_demonstrations = ["0001_01", "0001_02", "0001_03", "0001_04", "0001_05"]
		# list_of_demonstrations = ["0100_01", "0100_02", "0100_03", "0100_04", "0100_05"]

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
		print "\n----------- Combination #" + str(i) + " -------------\n"
		print combinations[i]
		mc = KinematicsClustering(DEBUG, list(combinations[i]), args.fname + str(i), log, vision_mode, feat_fname)
		all_metrics.append(mc.do_everything())

	print "----------- CALCULATING THE ODDS ------------"
	post_evaluation(all_metrics, log, args.fname, vision_mode)

	log.close()