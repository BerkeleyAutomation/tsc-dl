#!/usr/bin/env python

import pickle
import numpy as np
import argparse
import IPython
import itertools
import sys
from decimal import Decimal

import constants
import utils
import cluster_evaluation
import broken_barh
import time
import pruning

from sklearn import (mixture, neighbors, metrics, preprocessing)
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
mutual_info_score, homogeneity_score, completeness_score, recall_score, precision_score)

PATH_TO_FEATURES = constants.PATH_TO_DATA + constants.PROC_FEATURES_FOLDER

class TSCDL_singlemodal(object):
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
		self.map_frm2surgeme = utils.get_all_frame2surgeme_maps(self.list_of_demonstrations)
		self.trial = utils.hashcode() + fname

		# self.trial = fname
		self.cp_surgemes = []
		self.pruned_L1_clusters = []

		self.silhouette_score_global = None
		self.silhouette_score_weighted = None

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
			if (constants.TASK_NAME in ["000", "010", "011", "100"]):
				self.ALPHA_CP = constants.ALPHA_W_CP
		else:
			self.n_components_cp = constants.N_COMPONENTS_CP_W
			self.n_components_L1 = constants.N_COMPONENTS_L1_W
			self.temporal_window = constants.TEMPORAL_WINDOW_W
			self.representativeness = constants.PRUNING_FACTOR_W
			self.ALPHA_CP = constants.ALPHA_W_CP

		self.ALPHA_L1 = 0.4

		self.fit_GMM = False

		self.fit_DPGMM = True

		assert self.fit_DPGMM or self.fit_GMM == True

	def construct_features_visual(self):
		"""
		Loads visual features (saved as pickle files) and populates
		self.data_X dictionary
		"""

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
		"""
		Loads kinematic features (saved in text files) and populates
		self.data_X dictionary
		"""

		for demonstration in self.list_of_demonstrations:
			W = utils.sample_matrix(utils.get_kinematic_features(demonstration), sampling_rate = self.sr)
			scaler = preprocessing.StandardScaler().fit(W)
			self.data_X[demonstration] = scaler.transform(W)
			print "Kinematics ", demonstration, self.data_X[demonstration].shape

	def generate_transition_features(self):
		"""
		For each data point X(t), transition feature are created as follows:
		N(t) = X(t) + X(t+1) + .. + X(T), where T is self.temporal_window.
		"""
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

			start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
				+ demonstration + "_" + constants.CAMERA + ".p")

			for j in range(N.shape[0]):
				map_index2demonstration[i] = demonstration
				map_index2frm[i] = start + j * self.sr
				i += 1

			big_N = utils.safe_concatenate(big_N, N)

		print "Generating Changepoints. Fitting GMM/DP-GMM ..."

		if constants.REMOTE == 1:
			if self.fit_DPGMM:
				print "Init DPGMM"
				avg_len = int(big_N.shape[0]/len(self.list_of_demonstrations))
				DP_GMM_COMPONENTS = int(avg_len/constants.DPGMM_DIVISOR)
				print "L0", DP_GMM_COMPONENTS, "ALPHA: ", self.ALPHA_CP
				dpgmm = mixture.DPGMM(n_components = DP_GMM_COMPONENTS, covariance_type='diag', n_iter = 10000, alpha = self.ALPHA_CP, thresh= 1e-7)

			if self.fit_GMM:
				print "Init GMM"
				gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', n_iter=5000, thresh = 5e-5)

		if constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full', thresh = 0.01)

		else:
			gmm = mixture.GMM(n_components = self.n_components_cp, covariance_type='full')

		if self.fit_GMM:
			print "Fitting GMM"
			start = time.time()
			gmm.fit(big_N)
			end = time.time()
			print "GMM Time:", end - start

			Y_gmm = gmm.predict(big_N)
			print "L0: Clusters in GMM", len(set(Y_gmm))
			Y = Y_gmm

		if self.fit_DPGMM:
			print "Fitting DPGMM"
			start = time.time()
			dpgmm.fit(big_N)
			end = time.time()
			print "DPGMM Time:", end - start

			Y_dpgmm = dpgmm.predict(big_N)
			print "L0: Clusters in DP-GMM", len(set(Y_dpgmm))
			Y = Y_dpgmm

		for w in range(len(Y) - 1):

			if Y[w] != Y[w + 1]:
				change_pt = big_N[w][:self.X_dimension]
				self.append_cp_array(utils.reshape(change_pt))
				self.map_cp2frm[cp_index] = map_index2frm[w]
				self.map_cp2demonstrations[cp_index] = map_index2demonstration[w]
				self.list_of_cp.append(cp_index)

				cp_index += 1

		print "Done with generating change points, " + str(cp_index)

	def append_cp_array(self, cp):
		self.changepoints = utils.safe_concatenate(self.changepoints, cp)

	def save_cluster_metrics(self, points, predictions, key):
		"""
		Utility function calculates clustering metrics whenever
		clustering is performed in the algorithm.
		"""
		if key == 'level1':
			self.silhouette_score_global = metrics.silhouette_score(points, predictions, metric='euclidean')
			self.silhouette_score_weighted = utils.silhoutte_weighted(points, predictions)

		# dunn_scores = cluster_evaluation.dunn_index(points, predictions, means)

		dunn_scores = [0,0,0]

		if (dunn_scores[0] is not None) and (dunn_scores[1] is not None) and (dunn_scores[2] is not None):

			self.dunn_scores_1[key] = dunn_scores[0]
			self.dunn_scores_2[key] = dunn_scores[1]
			self.dunn_scores_3[key] = dunn_scores[2]

	def cluster_changepoints(self):
		"""
		Clusters changepoints specified in self.list_of_cp.
		"""

		print "Clustering changepoints..."
		print "L1 ", str(len(self.list_of_cp)/constants.DPGMM_DIVISOR_L1)," ALPHA: ", self.ALPHA_L1

		if constants.REMOTE == 1:
			if self.fit_DPGMM:
				dpgmm = mixture.DPGMM(n_components = int(len(self.list_of_cp)/constants.DPGMM_DIVISOR_L1), covariance_type='diag', n_iter = 10000, alpha = self.ALPHA_L1, thresh= 1e-4)
			if self.fit_GMM:
				gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', n_iter=5000, thresh = 0.01)
		elif constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', thresh = 0.01)
		else:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full')

		if self.fit_GMM:
			gmm.fit(self.changepoints)
			predictions_gmm = gmm.predict(self.changepoints)
			print "L1: Clusters in GMM",len(set(predictions_gmm))
			predictions = predictions_gmm

		if self.fit_DPGMM:
			predictions = []
			while True:
				print "Inside loop"
				dpgmm.fit(self.changepoints)
				predictions = dpgmm.predict(self.changepoints)
				if len(set(predictions)) > 1:
					break

			print "L1: Clusters in DP-GMM", len(set(predictions))

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
			weighted_data_representation = pruning.weighted_score(self.list_of_demonstrations, list(set(cluster_demonstrations)))

			print str(cluster) + ":  " + str(data_representation), " " + str(len(cluster_list_of_cp))
			print str(cluster) + ":w " + str(weighted_data_representation), " " + str(len(cluster_list_of_cp))

			val = weighted_data_representation if constants.WEIGHTED_PRUNING_MODE else data_representation

			if val <= self.representativeness:
				print "Pruned"
				new_cluster_list = cluster_list_of_cp[:]
				print "Pruned cluster"
				for cp in cluster_list_of_cp:
					self.list_of_cp.remove(cp)
					new_cluster_list.remove(cp)
				self.map_level1_cp[cluster] = new_cluster_list

		predictions = []
		filtered_changepoints = None
		inv_map = {v:k for k, v in constants.alphabet_map.items()}

		for cluster in self.map_level1_cp:
			cluster_list_of_cp = self.map_level1_cp[cluster]
			for cp in cluster_list_of_cp:
				predictions.append(inv_map[cluster])
				filtered_changepoints = utils.safe_concatenate(filtered_changepoints, utils.reshape(self.changepoints[cp]))

		predictions = np.array(predictions)

		self.save_cluster_metrics(filtered_changepoints, predictions, 'level1')

	def cluster_evaluation(self):

		for cp in self.list_of_cp:
			demonstration = self.map_cp2demonstrations[cp]
			frm = self.map_cp2frm[cp] + 2 * self.sr

			curr_surgeme = self.map_frm2surgeme[demonstration][frm]
			self.map_cp2surgemes[cp] = curr_surgeme

			ranges = sorted(utils.get_annotation_segments(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
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
		"""
		Computes metrics and returns as single dictionary.
		"""
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
			cp_all_data = (self.map_cp2frm[cp], self.map_cp2cluster[cp], self.map_cp2surgemetransitions[cp], self.map_cp2surgemes[cp])
			utils.dict_insert_list(self.map_cp2demonstrations[cp], cp_all_data, viz)

		data = [self.label_based_scores_1, self.silhouette_score_global, self.dunn_scores_1,
		self.dunn_scores_2, self.dunn_scores_3, viz, self.silhouette_score_weighted]

		return data

	def do_everything(self):

		if self.vision_mode:
			self.construct_features_visual()
		else:
			self.construct_features_kinematics()

		self.generate_transition_features()

		self.generate_change_points_2() #cluster over the full data set

		self.cluster_changepoints()

		self.cluster_pruning()

		self.cluster_evaluation()

		data = self.cluster_metrics()

		return data

def get_list_of_demo_combinations(list_of_demonstrations):
	"""
	For a given list_of_demonstrations with N demonstrations, function generates N combinations
	of N - 1 subsets of the inital dataset. This is used in the Jackknife
	estimate/Leave-one-out Cross Validation.
	"""
	iterator = itertools.combinations(list_of_demonstrations, len(list_of_demonstrations) - 1)
	demo_combinations = []
	while (1):
		try:
			demo_combinations.append(iterator.next())
		except StopIteration as e:
			break

	return demo_combinations

def post_evaluation_singlemodal(metrics, file, fname, vision_mode, list_of_demonstrations):

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
	silhouette_scores_global = []
	silhouette_scores_weighted = []

	dunn1_level_1 = []
	dunn2_level_1 = []
	dunn3_level_1 = []

	list_of_frms = {}

	for elem in metrics:

		mutual_information_1.append(elem[0]["mutual_info_score"])
		normalized_mutual_information_1.append(elem[0]["normalized_mutual_info_score"])
		adjusted_mutual_information_1.append(elem[0]["adjusted_mutual_info_score"])
		homogeneity_1.append(elem[0]["homogeneity_score"])

		silhouette_scores_global.append(elem[1])
		silhouette_scores_weighted.append(elem[6])

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
	utils.print_and_write_2("silhouette_scores_global", np.mean(silhouette_scores_global), np.std(silhouette_scores_global), file)
	utils.print_and_write_2("silhouette_scores_weighted", np.mean(silhouette_scores_weighted), np.std(silhouette_scores_weighted), file)

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

	data_cache = {}

	for demonstration in list_of_demonstrations:
		try:
			list_of_frms_demonstration = list_of_frms[demonstration]

			data = {}

			for i in range(len(list_of_frms_demonstration)):
				data[i] = [elem[0] for elem in list_of_frms_demonstration[i]]

			dtw_score, normalized_dtw_score, length, labels_manual_d, colors_manual_d, labels_automatic_d, colors_automatic_d = broken_barh.plot_broken_barh(demonstration, data,
				constants.PATH_TO_CLUSTERING_RESULTS + demonstration +"_" + fname + ".jpg", T)

			list_of_dtw_values.append(dtw_score)
			list_of_norm_dtw_values.append(normalized_dtw_score)
			list_of_lengths.append(length)
		
			# Inserting manual and annotations labels into data struct before dumping as pickle file
			cache_entry = {}
			cache_entry['changepoints'] = list_of_frms_demonstration
			cache_entry['plot_labels_manual'] = labels_manual_d
			cache_entry['plot_colors_manual'] = colors_manual_d
			cache_entry['plot_labels_automatic'] = labels_automatic_d
			cache_entry['plot_colors_automatic'] = colors_automatic_d
			data_cache[demonstration] = cache_entry

		except:
			print demonstration
			pass

	utils.print_and_write_2("dtw_score", np.mean(list_of_dtw_values), np.std(list_of_dtw_values), file)
	utils.print_and_write_2("dtw_score_normalized", np.mean(list_of_norm_dtw_values), np.std(list_of_norm_dtw_values), file)
	utils.print_and_write(str(list_of_lengths), file)

	pickle.dump(data_cache, open(constants.PATH_TO_CLUSTERING_RESULTS + fname + "_.p", "wb"))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	argparser.add_argument("--visual", help = "Name of pickle file", default = False)
	argparser.add_argument("fname", help = "Name of experiment", default = 4)
	args = argparser.parse_args()

	if args.debug == 'y':
		DEBUG = True
		list_of_demonstrations = ['Suturing_E001','Suturing_E002']
	else:
		DEBUG = False

		list_of_demonstrations = constants.config.get("list_of_demonstrations")

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
		mc = TSCDL_singlemodal(DEBUG, list(combinations[i]), args.fname + str(i), log, vision_mode, feat_fname)
		all_metrics.append(mc.do_everything())

	print "----------- CALCULATING THE ODDS ------------"
	post_evaluation_singlemodal(all_metrics, log, args.fname, vision_mode, list_of_demonstrations)

	log.close()