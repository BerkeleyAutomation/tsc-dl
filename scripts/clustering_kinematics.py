#!/usr/bin/env python

import pickle
import numpy as np
import argparse
import IPython
import itertools

import constants
import parser
import utils
import cluster_evaluation
import featurization

from sklearn import (mixture, neighbors, metrics)
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
mutual_info_score, homogeneity_score, completeness_score, recall_score, precision_score)

PATH_TO_FEATURES = constants.PATH_TO_SUTURING_DATA + constants.PROC_FEATURES_FOLDER

class KinematicsClustering():
	def __init__(self, DEBUG, list_of_demonstrations, fname):
		self.list_of_demonstrations = list_of_demonstrations
		self.data_X = {}
		self.data_X_size = {}
		self.data_N = {}

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

		self.sr = 3

	def construct_features(self):

		for demonstration in self.list_of_demonstrations:
			self.data_X[demonstration] = featurization.get_kinematic_features[demonstration]

	def generate_transition_features(self):
		print "Generating Transition Features"

		for demonstration in self.list_of_demonstrations:

			X = self.data_X[demonstration]
			self.data_X_size[demonstration] = X.shape[1]
			T = X.shape[0]
			N = utils.reshape(np.concatenate((X[0], X[1]), axis = 1))

			for t in range(1, T - 1):

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
	
			self.save_cluster_metrics(N, Y, gmm.means_, 'cpts_' + demonstration, gmm)

			for i in range(len(Y) - 1):

				if Y[i] != Y[i + 1]:

					change_pt = N[i][size_of_X:]
					print N.shape, change_pt.shape
					self.append_cp_array(utils.reshape(change_pt))
					self.map_cp2frm[cp_index] = i
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

		print "Level1 : Clustering changepoints in Z(t)"

		gmm = mixture.GMM(n_components = 10, covariance_type='full')
		gmm.fit(self.changepoints)

		predictions = gmm.predict(self.changepoints)

		self.save_cluster_metrics(self.changepoints, predictions, gmm.means_, 'level1', gmm)

		for i in range(len(predictions)):
			label = constants.alphabet_map[Y[i] + 1]
			self.map_cp2cluster[i] = label
			utils.dict_insert_list(label, i, self.map_level1_cp)
			demonstration = self.map_cp2demonstrations[i]
			frm = self.map_cp2frm[i]
			surgeme = self.map_frm2surgeme[demonstration][frm]

			print("%3d   %s   %s   %3d   %3d\n" % (cp, label, demonstration, frm, surgeme))

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
			surgeme_label = self.map_frm2surgeme[self.map_cp2demonstrations[cp]][self.map_cp2frm[cp]]
			self.map_cp2surgemes[cp] = surgeme_label
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

		print('\n\n ---Confusion Matrix--- \n\n')
		print(confusion_matrix)

		confusion_matrix = ""
		for L1_cluster in final_clusters:
			confusion_matrix = confusion_matrix + "\n" + L1_cluster + "   "
			for surgeme in self.cp_surgemes:
				# confusion_matrix += str(float("{0:.2f}".format(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])))) + "   "
				confusion_matrix += str(round(Decimal(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])), 2)) + "   "
			confusion_matrix += '\n'

		print confusion_matrix
		print(confusion_matrix)
		print("\n\n ---Surgeme Count--- \n\n")
		print(repr(surgeme_count))
		print("\n\n")

	def prepare_labels(self):
		labels_pred = []
		labels_true = []

		for cp in self.list_of_cp:
			labels_true.append(self.map_cp2surgemes[cp])
			labels_pred.append(self.map_cp2cluster[cp])

		assert len(labels_true) == len(labels_pred)

		return labels_true, labels_pred

	def cluster_metrics(self):
		labels_true, labels_pred = self.prepare_labels()

		metric_funcs = [adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, 
		mutual_info_score, homogeneity_score, completeness_score]

		# ------ Label-based metrics ------
		print("\n\nPred= L1 Labels     Metric\n\n")

		for metric in metric_funcs:

			score_1 = round(Decimal(metric(labels_true, labels_pred)), 2)
			key =  repr(metric).split()[1]
			self.label_based_scores_1[key] = score_1

			print("%3.3f        %s\n" % (score_1, key))

		print("\nSilhoutte scores\n")

		# ------ Silhouette Scores ------
		for layer in sorted(self.silhouette_scores):
			score = self.silhouette_scores[layer]
			print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		print("\nDunn Scores1\n")

		# ------ Dunn Scores # 1------
		for layer in sorted(self.dunn_scores_1):
			score = self.dunn_scores_1[layer]
			print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		print("\nDunn Scores2\n")

		# ------ Dunn Scores # 2------
		for layer in sorted(self.dunn_scores_2):
			score = self.dunn_scores_2[layer]
			print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		print("\nDunn Scores3\n")

		# ------ Dunn Scores #3 ------
		for layer in sorted(self.dunn_scores_3):
			score = self.dunn_scores_3[layer]
			print("%3.3f        %s\n" % (round(Decimal(score), 2), layer))

		data = [self.label_based_scores_1, self.silhouette_scores, self.dunn_scores_1,
		self.dunn_scores_2, self.dunn_scores_3]

		pickle.dump(data, open(self.metrics_picklefile, "wb"))

		return data

	def do_everything(self):

		self.construct_features()

		self.generate_transition_features()

		self.generate_change_points()

		self.cluster_changepoints()

		self.cluster_evaluation()

		data = self.cluster_metrics()

		return data

def print_and_write(metric, mean, std, file):
	print("\n%1.3f  %1.3f  %s\n" % (mean, std, metric))
	file.write("\n%1.3f  %1.3f  %s\n" % (mean, std, metric))

def get_list_of_demo_combinations(list_of_demonstrations):
	iterator = itertools.combinations(list_of_demonstrations, len(list_of_demonstrations) - 1)
	demo_combinations = []

	while (1):
		try:
			demo_combinations.append(iterator.next())
		except StopIteration as e:
			break

	return demo_combinations

def parse_metrics(metrics, fname):

	mutual_information_1 = homogeneity_1 = []

	silhoutte_level_1 = dunn1_level_1 = dunn2_level_1 = dunn3_level_1 = []

	for elem in metrics:
		mutual_information_1.append(elem[0]["mutual_info_score"])
		homogeneity_1.append(elem[0]["homogeneity_score"])

		silhoutte_level_1.append(elem[2]["level1"])
		dunn1_level_1.append(elem[3]["level1"])
		dunn2_level_1.append(elem[4]["level1"])
		dunn3_level_1.append(elem[5]["level1"])

	file = open(constants.PATH_TO_CLUSTERING_RESULTS + fname + ".txt", "wb")

	print_and_write("mutual_info", np.mean(mutual_information_1), np.std(mutual_information_1), file)
	print_and_write("homogeneity", np.mean(homogeneity_1), np.std(homogeneity_1), file)
	print_and_write("silhoutte_level_1", np.mean(silhoutte_level_1), np.std(silhoutte_level_1), file)

	print_and_write("dunn1", np.mean(dunn1_level_1), np.std(dunn1_level_1), file)
	print_and_write("dunn2", np.mean(dunn2_level_1), np.std(dunn2_level_1), file)
	print_and_write("dunn3", np.mean(dunn3_level_1), np.std(dunn3_level_1), file)

	file.close()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	argparser.add_argument("fname", help = "Pickle file of visual features", default = 4)
	args = argparser.parse_args()

	if args.debug == 'y':
		DEBUG = True
		list_of_demonstrations = ['Suturing_E001','Suturing_E002']
	else:
		DEBUG = False
		list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']
	
	combinations = get_list_of_demo_combinations(list_of_demonstrations)

	all_metrics = []
	for elem in combinations:	
		mc = KinematicsClustering(DEBUG, list(elem), args.fname)
		all_metrics.append(mc.do_everything())

	print "----------- CALCULATING THE ODDS ------------"
	parse_metrics(all_metrics, args.fname)