#!/usr/bin/env python

import pickle
import numpy as np
import IPython
import argparse
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors

import constants
import utils
from clustering import TSCDL_multimodal, get_list_of_demo_combinations, post_evaluation_multimodal
from clustering_kinematics import TSCDL_singlemodal

from sklearn import (mixture, preprocessing, neighbors, metrics, cross_decomposition)
from decimal import Decimal
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
mutual_info_score, homogeneity_score, completeness_score, recall_score, precision_score)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

class TSCDL_manual_multimodal(TSCDL_multimodal):

	def __init__(self, DEBUG, list_of_demonstrations, featfile, trialname):
		super(TSCDL_manual_multimodal, self).__init__(DEBUG, list_of_demonstrations, featfile, trialname)

		# This ensures no pruning
		self.representativeness = -1.0

	def generate_change_points_2(self):
		"""
		Generates changespoints by clustering across demonstrations.
		"""
		cp_index = 0

		for demonstration in self.list_of_demonstrations:
			W = self.data_W[demonstration]
			Z = self.data_Z[demonstration]

			PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
			annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
			manual_labels = utils.get_chronological_sequences(annotations)
			start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)

			for elem in manual_labels:
				frm = elem[1]
				change_pt_W = W[(frm - start)/self.sr]
				change_pt_Z = Z[(frm - start)/self.sr]
				change_pt = utils.safe_concatenate(change_pt_W, change_pt_Z)

				self.append_cp_array(change_pt)
				self.map_cp2demonstrations[cp_index] = demonstration
				self.map_cp2frm[cp_index] = frm
				self.list_of_cp.append(cp_index)
				cp_index += 1


class TSCDL_manual_singlemodal(TSCDL_singlemodal):

	def __init__(self, DEBUG, list_of_demonstrations, fname, log, vision_mode = False, feat_fname = None):
		super(TSCDL_manual_singlemodal, self).__init__(DEBUG, list_of_demonstrations, fname, log, vision_mode, feat_fname)

		# This ensures no pruning
		self.representativeness = -1.0

	def generate_change_points_2(self):
		"""
		Generates changespoints by clustering across demonstrations.
		"""
		cp_index = 0

		for demonstration in self.list_of_demonstrations:
			X = self.data_X[demonstration]

			PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
			annotations = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
			manual_labels = utils.get_chronological_sequences(annotations)
			start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)

			for elem in manual_labels:
				frm = elem[1]
				change_pt = X[(frm - start)/self.sr]

				self.append_cp_array(utils.reshape(change_pt))
				self.map_cp2demonstrations[cp_index] = demonstration
				self.map_cp2frm[cp_index] = frm
				self.list_of_cp.append(cp_index)
				cp_index += 1

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("feat_fname", help = "Pickle file of visual features", default = 4)
	argparser.add_argument("fname", help = "Pickle file of visual features", default = 4)
	args = argparser.parse_args()

	combinations = get_list_of_demo_combinations(list_of_demonstrations)

	i = 1
	all_metrics = []

	for elem in combinations:	
		print "---- k-Fold Cross Validation, Run "+ str(i) + " out of " + str(len(combinations)) + " ----"
		mc = TSCDL_manual_multimodal(False, list(elem), args.feat_fname, args.fname)
		all_metrics.append(mc.do_everything())
		i += 1

	print "----------- CALCULATING THE ODDS ------------"
	post_evaluation_multimodal(all_metrics, args.fname, list_of_demonstrations, args.feat_fname)
