#!/usr/bin/env python

import pickle
import sys
import os
import argparse
import sys
import IPython

import constants
import utils
from clustering import TSCDL_multimodal, post_evaluation_multimodal, get_list_of_demo_combinations
from clustering_kinematics import TSCDL_singlemodal, post_evaluation_singlemodal
from broken_barh import plot_broken_barh_from_pickle
from jaccard import TSCDL_manual_multimodal, TSCDL_manual_singlemodal

# One script to rule all other scripts
# Adithya, 2016

def check_visual_features(args):
	"""
	Function to check for visual features
	"""
	if not args.visual_feature:
		print "ERROR: Please specify visual feature: --visual_feature <feature file>.p"
		os.remove(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt")
		sys.exit()

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("mode", help = "python tscdl.py Z _test_ --visual_feature 5_PCA.p Choose from 1) W (Kinematics only),\n 2) Z (Visual features only),\n 3) ZW (Kinematics+Vision),\n 4) plot,\n 5) control-W,\n 6) control-Z,\n 7) control-ZW")
	argparser.add_argument("output_fname", help = "Pickle file of visual features")
	argparser.add_argument("--visual_feature", help = "Pickle file of visual features")
	argparser.add_argument("--W", help = "Cache of data running TSCDL on Kinematics only")
	argparser.add_argument("--Z", help = "Cache of data running TSCDL on Vision only")
	argparser.add_argument("--ZW", help = "Cache of data running TSCDL on Both kinematics and vision")
	args = argparser.parse_args()

	list_of_demonstrations = constants.config.get("list_of_demonstrations")
	combinations = get_list_of_demo_combinations(list_of_demonstrations)
	all_metrics = []

	if args.mode == "ZW":
		check_visual_features(args)

		file = open(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt", "wb")
		for i, elem in enumerate(combinations):
			print "---- Kinematics + Vision Combination #"+ str(i + 1) + "----"
			mc = TSCDL_multimodal(False, list(elem), args.visual_feature, args.output_fname)
			all_metrics.append(mc.do_everything())

		print "----------- CALCULATING THE ODDS ------------"
		post_evaluation_multimodal(all_metrics, file, args.output_fname, list_of_demonstrations, args.visual_feature)
		file.close()

	elif args.mode == "W":

		file = open(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt", "wb")
		for i, elem in enumerate(combinations):
			utils.print_and_write("\n----------- Kinematics-only Combination #" + str(i + 1) + " -------------\n", file)
			print "\n----------- Kinematics-only Combination #" + str(i + 1) + " -------------\n"
			print elem
			mc = TSCDL_singlemodal(False, list(elem), args.output_fname + str(i), file, False, None)
			all_metrics.append(mc.do_everything())

		print "----------- CALCULATING THE ODDS ------------"
		post_evaluation_singlemodal(all_metrics, file, args.output_fname, False, list_of_demonstrations)
		file.close()

	elif args.mode == "Z":
		check_visual_features(args)

		file = open(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt", "wb")
		for i, elem in enumerate(combinations):
			utils.print_and_write("\n----------- Vision-only Combination #" + str(i + 1) + " -------------\n", file)
			print "\n----------- Vision-only Combination #" + str(i + 1) + " -------------\n"
			print elem
			mc = TSCDL_singlemodal(False, list(elem), args.output_fname + str(i), file, True, args.visual_feature)
			all_metrics.append(mc.do_everything())

		print "----------- CALCULATING THE ODDS ------------"
		post_evaluation_singlemodal(all_metrics, file, args.output_fname, True, list_of_demonstrations)
		file.close()

	elif args.mode == "plot":
		if not (args.Z and args.ZW and args.W):
			print "ERROR: Please specify pickle files for all TSCDL experiments"
			sys.exit()

		W = pickle.load(open(args.W, "rb"))
		Z = pickle.load(open(args.Z, "rb"))
		ZW = pickle.load(open(args.ZW, "rb"))

		assert len(W.keys()) == len(Z.keys()) == len(ZW.keys())
		for demonstration in W.keys():
			labels_manual = W[demonstration]['plot_labels_manual']
			colors_manual = W[demonstration]['plot_colors_manual']
			labels_W = W[demonstration]['plot_labels_automatic']
			colors_W = W[demonstration]['plot_colors_automatic']
			labels_Z = Z[demonstration]['plot_labels_automatic']
			colors_Z = Z[demonstration]['plot_colors_automatic']
			labels_ZW = ZW[demonstration]['plot_labels_automatic']
			colors_ZW = ZW[demonstration]['plot_colors_automatic']


			plot_broken_barh_from_pickle(demonstration, args.output_fname + "_" + demonstration, labels_manual, colors_manual,
				labels_W, colors_W, labels_Z, colors_Z, labels_ZW, colors_ZW)

	if args.mode == "control-ZW":
		check_visual_features(args)
	
		file = open(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt", "wb")
		for i, elem in enumerate(combinations):
			print "---- CONTROL: Kinematics + Vision Combination #"+ str(i + 1) + "----"
			mc = TSCDL_manual_multimodal(False, list(elem), args.visual_feature, args.output_fname)
			all_metrics.append(mc.do_everything())

		print "----------- CALCULATING THE ODDS ------------"
		post_evaluation_multimodal(all_metrics, file, args.output_fname, list_of_demonstrations, args.visual_feature)
		file.close()

	elif args.mode == "control-W":

		file = open(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt", "wb")
		for i, elem in enumerate(combinations):
			utils.print_and_write("\n----------- Kinematics-only Combination #" + str(i + 1) + " -------------\n", file)
			print "\n----------- CONTROL: Kinematics-only Combination #" + str(i + 1) + " -------------\n"
			print elem
			mc = TSCDL_manual_singlemodal(False, list(elem), args.output_fname + str(i), file, False, None)
			all_metrics.append(mc.do_everything())

		print "----------- CALCULATING THE ODDS ------------"
		post_evaluation_singlemodal(all_metrics, file, args.output_fname, False, list_of_demonstrations)
		file.close()

	elif args.mode == "control-Z":
		check_visual_features(args)

		file = open(constants.PATH_TO_CLUSTERING_RESULTS + args.output_fname + ".txt", "wb")
		for i, elem in enumerate(combinations):
			utils.print_and_write("\n----------- Vision-only Combination #" + str(i + 1) + " -------------\n", file)
			print "\n----------- CONTROL: Vision-only Combination #" + str(i + 1) + " -------------\n"
			print elem
			mc = TSCDL_manual_singlemodal(False, list(elem), args.output_fname + str(i), file, True, args.visual_feature)
			all_metrics.append(mc.do_everything())

		print "----------- CALCULATING THE ODDS ------------"
		post_evaluation_singlemodal(all_metrics, file, args.output_fname, True, list_of_demonstrations)
		file.close()
	else:
		print "ERROR: Please specify a valid mode"
