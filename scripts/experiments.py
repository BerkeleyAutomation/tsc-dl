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
import clustering
import clustering_kinematics

from clustering import MilestonesClustering
from clustering_kinematics import KinematicsClustering

PATH_TO_FEATURES = constants.PATH_TO_DATA + constants.PROC_FEATURES_FOLDER

def evaluate_multi_modal(metrics, file):

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

	return list_of_frms

def evaluate_single_modal(metrics, file):
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

	return list_of_frms

def post_evaluation_all(metrics_W, metrics_Z, metrics_ZW, file, fname, list_of_demonstrations):

	utils.print_and_write("\nXXXXXXXXXXXX Metric for Kinematics W XXXXXXXXXXXX\n", file)
	list_of_frms_W = evaluate_single_modal(metrics_W, file)

	utils.print_and_write("\nXXXXXXXXXXXX Metric for Visual Z XXXXXXXXXXXX\n", file)
	list_of_frms_Z = evaluate_single_modal(metrics_Z, file)

	utils.print_and_write("\nXXXXXXXXXXXX Metric for Kinematics + Visual ZW XXXXXXXXXXXX\n", file)
	list_of_frms_ZW = evaluate_multi_modal(metrics_ZW, file)

	list_of_dtw_values_W = []
	list_of_dtw_values_Z = []
	list_of_dtw_values_ZW = []
	list_of_dtw_values_W_normalized = []
	list_of_dtw_values_Z_normalized = []
	list_of_dtw_values_ZW_normalized = []
	list_of_lengths = []

	for demonstration in list_of_demonstrations:
		list_of_frms_demonstration_W = list_of_frms_W[demonstration]
		list_of_frms_demonstration_Z = list_of_frms_Z[demonstration]
		list_of_frms_demonstration_ZW = list_of_frms_ZW[demonstration]

		assert len(list_of_frms_demonstration_W) == len(list_of_demonstrations) - 1
		assert len(list_of_frms_demonstration_Z) == len(list_of_demonstrations) - 1
		assert len(list_of_frms_demonstration_ZW) == len(list_of_demonstrations) - 1
		data_W = {}
		data_Z = {}
		data_ZW = {}

		for i in range(len(list_of_frms_demonstration_W)):
			data_W[i] = list_of_frms_demonstration_W[0]
			data_Z[i] = list_of_frms_demonstration_Z[0]
			data_ZW[i] = list_of_frms_demonstration_ZW[0]

		save_fig = constants.PATH_TO_CLUSTERING_RESULTS + demonstration + "_" + fname + "_A.jpg"
		save_fig2 = constants.PATH_TO_CLUSTERING_RESULTS + demonstration + "_" + fname + "_B.jpg"

		dtw_score_W, dtw_score_Z, dtw_score_ZW, dtw_score_W_normalized, dtw_score_Z_normalized, dtw_score_ZW_normalized, length = broken_barh.plot_broken_barh_all(demonstration,
			data_W, data_Z, data_ZW, save_fig, save_fig2)
		list_of_dtw_values_W.append(dtw_score_W)
		list_of_dtw_values_Z.append(dtw_score_Z)
		list_of_dtw_values_ZW.append(dtw_score_ZW)
		list_of_dtw_values_W_normalized.append(dtw_score_W_normalized)
		list_of_dtw_values_Z_normalized.append(dtw_score_Z_normalized)
		list_of_dtw_values_ZW_normalized.append(dtw_score_ZW_normalized)
		list_of_lengths.append(length)

	utils.print_and_write_2("dtw_score_W", np.mean(list_of_dtw_values_W), np.std(list_of_dtw_values_W), file)
	utils.print_and_write_2("dtw_score_Z", np.mean(list_of_dtw_values_Z), np.std(list_of_dtw_values_Z), file)
	utils.print_and_write_2("dtw_score_ZW", np.mean(list_of_dtw_values_ZW), np.std(list_of_dtw_values_ZW), file)
	utils.print_and_write_2("dtw_score_W_normalized", np.mean(list_of_dtw_values_W_normalized), np.std(list_of_dtw_values_W_normalized), file)
	utils.print_and_write_2("dtw_score_Z_normalized", np.mean(list_of_dtw_values_Z_normalized), np.std(list_of_dtw_values_Z_normalized), file)
	utils.print_and_write_2("dtw_score_ZW_normalized", np.mean(list_of_dtw_values_ZW_normalized), np.std(list_of_dtw_values_ZW_normalized), file)
	utils.print_and_write(str(list_of_lengths), file)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("visual", help = "Specify visual features, e.g. 4_PCA.p", default = False)
	argparser.add_argument("fname", help = "File name", default = 4)
	args = argparser.parse_args()

	# list_of_demonstrations = ["100_01", "100_02", "100_03", "100_04", "100_05"]
	# list_of_demonstrations = ["011_01", "011_02", "011_03", "011_04", "011_05"]
	list_of_demonstrations = ["plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

	# list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']

	combinations = clustering.get_list_of_demo_combinations(list_of_demonstrations)

	feat_fname = args.visual

	metrics_W = []
	metrics_Z = []
	metrics_ZW = []

	log = open(constants.PATH_TO_CLUSTERING_RESULTS + args.fname + ".txt", "wb")

	utils.print_and_write("\nXXXXXXXXXXXX Kinematics W XXXXXXXXXXXX\n", log)
	print "\nXXXXXXXXXXXX Kinematics W XXXXXXXXXXXX\n"

	for i in range(len(combinations)):
		utils.print_and_write("\n----------- Combination #" + str(i) + " -------------\n", log)
		print "\n----------- Combination #" + str(i) + " -------------\n"
		print combinations[i]
		mc = KinematicsClustering(False, list(combinations[i]), args.fname + str(i), log, False, feat_fname)
		metrics_W.append(mc.do_everything())

	utils.print_and_write("\nXXXXXXXXXXXX Visual Z XXXXXXXXXXXX\n", log)
	print "\nXXXXXXXXXXXX Visual Z XXXXXXXXXXXX\n"

	for i in range(len(combinations)):
		utils.print_and_write("\n----------- Combination #" + str(i) + " -------------\n", log)
		print "\n----------- Combination #" + str(i) + " -------------\n"
		print combinations[i]
		mc = KinematicsClustering(False, list(combinations[i]), args.fname + str(i), log, True, feat_fname)
		metrics_Z.append(mc.do_everything())

	utils.print_and_write("\nXXXXXXXXXXXX Kinematics + Vision ZW XXXXXXXXXXXX\n", log)
	print "\nXXXXXXXXXXXX Kinematics + Visual ZW XXXXXXXXXXXX\n"

	for i in range(len(combinations)):
		print "---- k-Fold Cross Validation, Run "+ str(i) + " out of " + str(len(combinations)) + " ----"
		mc = MilestonesClustering(False, list(combinations[i]), feat_fname, args.fname)
		metrics_ZW.append(mc.do_everything())

	post_evaluation_all(metrics_W, metrics_Z, metrics_ZW, log, args.fname, list_of_demonstrations)

	log.close()