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
from clustering import TSCDL_multimodal, get_list_of_demo_combinations, post_evaluation

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

class TSCDL_manual(TSCDL_multimodal):

	def cluster_changepoints_level1(self):

		print "Level1 : Clustering changepoints in Z(t)"

		if constants.REMOTE == 1:
			if self.fit_DPGMM:
				print "DPGMM L1 - start"
				# Previously, when L0 was GMM, alpha = 0.4
				print "L1 ", str(len(self.list_of_cp)/constants.DPGMM_DIVISOR_L1), " ALPHA ", 10
				dpgmm = mixture.DPGMM(n_components = int(len(self.list_of_cp)/6), covariance_type='diag', n_iter = 1000, alpha = 0.0005, thresh= 1e-7)
				print "DPGMM L1 - end"

			if self.fit_GMM:
				gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full', n_iter=1000, thresh = 5e-5)
				print "GMM L1 - end"
		elif constants.REMOTE == 2:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full')
		else:
			gmm = mixture.GMM(n_components = self.n_components_L1, covariance_type='full')

		if self.fit_GMM:
			gmm.fit(self.change_pts_Z)
			Y_gmm = gmm.predict(self.change_pts_Z)
			Y = Y_gmm


		if self.fit_DPGMM:
			Y_dpgmm = []
			i = 0

			while True:
				print "In DPGMM Fit loop"
				dpgmm.fit(self.change_pts_Z)
				Y_dpgmm = dpgmm.predict(self.change_pts_Z)
				if len(set(Y_dpgmm)) > 1:
					break
				i += 1
				if i > 100:
					break

		self.save_cluster_metrics(self.change_pts_Z, Y_dpgmm, 'level1')

		for i in range(len(Y_dpgmm)):
			label = constants.alphabet_map[Y_dpgmm[i] + 1]
			self.map_cp2level1[i] = label
			utils.dict_insert_list(label, i, self.map_level12cp)

		self.generate_l2_cluster_matrices()

		# color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

		clf = dpgmm
		Y_ = Y_dpgmm
		cmap = get_cmap(len(set(Y_)))
		X = self.change_pts_Z
		X = utils.pca(X)
		x_min, x_max = np.min(X, 0), np.max(X, 0)
		X = (X - x_min) / (x_max - x_min)

		# splot = plt.subplot(2, 1, 1)
		for i, (mean, covar) in enumerate(zip(clf.means_, clf._get_covars())):
			v, w = linalg.eigh(covar)
			u = w[0] / linalg.norm(w[0])
			# as the DP will not use every component it has access to
			# unless it needs it, we shouldn't plot the redundant components.
			if not np.any(Y_ == i):
				continue
			plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 20, color=cmap(i))

			# Plot an ellipse to show the Gaussian component
			# angle = np.arctan(u[1] / u[0])
			# angle = 180 * angle / np.pi  # convert to degrees
			# ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
			# ell.set_clip_box(splot.bbox)
			# ell.set_alpha(0.5)
			# splot.add_artist(ell)

		# plt.xlim(0.8, 1)
		# plt.ylim(0.8, 1)
		# plt.xticks(())
		# plt.yticks(())

		# plt.show()

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
		list_of_demonstrations = constants.config.get("list_of_demonstrations")

	combinations = get_list_of_demo_combinations(list_of_demonstrations)

	i = 1
	all_metrics = []

	for elem in combinations:	
		print "---- k-Fold Cross Validation, Run "+ str(i) + " out of " + str(len(combinations)) + " ----"
		mc = TSCDL_manual(DEBUG, list(elem), args.feat_fname, args.fname)
		all_metrics.append(mc.do_everything())
		i += 1

	print "----------- CALCULATING THE ODDS ------------"
	post_evaluation(all_metrics, args.fname, list_of_demonstrations, args.feat_fname)
