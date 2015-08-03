#!/usr/bin/env python

import IPython
import pickle
import numpy as np
import random
import os

import constants
import parser
import utils

from sklearn import (mixture, preprocessing, neighbors)
from decimal import Decimal

class MilestonesClustering():
	def __init__(self):
		# self.list_of_demonstrations = parser.generate_list_of_videos(constants.PATH_TO_SUTURING_DATA + constants.CONFIG_FILE)
		self.list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']
		self.data_X = {}
		self.data_N = {}

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
		self.map_cp2milestone = {}
		self.l2_cluster_matrices = {}
		self.map_frm2surgeme = parser.get_all_frame2surgeme_maps(self.list_of_demonstrations)
		self.trial = utils.hashcode()

		self.pruned_L1_clusters = []
		self.pruned_L2_clusters = []

	def get_kinematic_features(self, demonstration, sampling_rate = 1):
		return parser.parse_kinematics(constants.PATH_TO_SUTURING_KINEMATICS, constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_capture2.p",
			demonstration + ".txt", sampling_rate = sampling_rate)

	def get_visual_features(self, demonstration, PC = 100):
		layer = 'conv4'

		Z = pickle.load(open(constants.PATH_TO_SUTURING_DATA + constants.ALEXNET_FEATURES_FOLDER + layer
			+ "_alexnet_" + demonstration + "_capture2.p", "rb"))

		return utils.pca(Z.astype(np.float), PC)

	def construct_features(self):
		for demonstration in self.list_of_demonstrations:
			print "Parsing Kinematics " + demonstration
			W = self.get_kinematic_features(demonstration)
			# scaler = preprocessing.StandardScaler().fit(W)
			# W = scaler.transform(W)

			print "Parsing visual features " + demonstration
			Z = self.get_visual_features(demonstration, PC = 100)

			X = np.concatenate((W, Z), axis = 1)
			self.data_X[demonstration] = X

	def generate_transition_features(self):
		print "Generating Transition Features"

		for demonstration in self.list_of_demonstrations:

			X = self.data_X[demonstration]
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

			map_cp2frm = {}
			for i in range(len(Y) - 1):

				if Y[i] != Y[i + 1]:

					change_pt = N[i][N.shape[1]/2:]
					self.append_cp_array(change_pt)
					self.map_cp2frm[cp_index] = i + 1
					self.map_cp2demonstrations[cp_index] = demonstration
					self.list_of_cp.append(cp_index)

					cp_index += 1

	def append_cp_array(self, cp):

		if self.change_pts is None:
			self.change_pts = utils.reshape(cp)
			self.change_pts_W = utils.reshape(cp[:38])
			self.change_pts_Z = utils.reshape(cp[38:])

		else:
			self.change_pts = np.concatenate((self.change_pts, utils.reshape(cp)), axis = 0)
			self.change_pts_W = np.concatenate((self.change_pts_W, utils.reshape(cp[:38])), axis = 0)
			self.change_pts_Z = np.concatenate((self.change_pts_Z, utils.reshape(cp[38:])), axis = 0)

	def cluster_changepoints_level1(self):

		print "Level1 : Clustering changepoints in Z(t)"

		gmm = mixture.GMM(n_components = 10, covariance_type='full')
		gmm.fit(self.change_pts_Z)

		Y = gmm.predict(self.change_pts_Z)

		for i in range(len(Y)):
			label = constants.alphabet_map[Y[i] + 1]
			self.map_cp_level1[i] = label
			utils.dict_insert_list(label, i, self.map_level1_cp)

		self.generate_l2_cluster_matrices()

	def generate_l2_cluster_matrices(self):

		for key in self.map_level1_cp.keys():

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

		print "Level2 : Clustering changepoints in W(t)"

		mkdir_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial
		os.mkdir(mkdir_path)

		# To put frames of milestones
		os.mkdir(mkdir_path + "/" + "milestones")

		file = open(mkdir_path + "/" + self.trial + "clustering.txt", "wb")
		line = "L1 Cluster   L2 Cluster   Demonstration   Frame#  CP#   Surgeme\n"
		file.write(line)

		for key in self.map_level1_cp.keys():
			mkdir_l1_cluster = mkdir_path + "/" + key

			list_of_cp = self.map_level1_cp[key]

			if self.check_pruning_condition(list_of_cp):
				continue

			os.mkdir(mkdir_l1_cluster)

		for key in self.map_level1_cp.keys():
			matrix = self.l2_cluster_matrices[key]
			list_of_cp = self.map_level1_cp[key]

			if self.check_pruning_condition(list_of_cp):
				continue

			n_components = 3
			gmm = mixture.GMM(n_components = n_components, covariance_type='full')

			try:
				gmm.fit(matrix)

			except ValueError as e:

				self.pruned_L1_clusters.append(key)
				print "Unable to fit GMM!"
				continue

			Y = gmm.predict(matrix)

			for i in range(len(Y)):

				cp = list_of_cp[i]
				l1_cluster = key
				l2_cluster = Y[i]
				milestone = l1_cluster + str(l2_cluster)
				demonstration = self.map_cp2demonstrations[cp]
				frm = self.map_cp2frm[cp]
				surgeme = self.map_frm2surgeme[demonstration][frm]

				self.map_cp2milestone[cp] = milestone

				file.write("%s             %3d         %s   %3d   %3d    %3d\n" % (l1_cluster, l2_cluster, demonstration, frm, cp, surgeme))

				self.copy_frames(demonstration, frm, str(l1_cluster), str(l2_cluster), surgeme)

			self.copy_milestone_frames(matrix, list_of_cp, gmm)

	def copy_milestone_frames(self, matrix, list_of_cp, gmm):
		neigh = neighbors.KNeighborsClassifier(n_neighbors = 1)
		neigh.fit(matrix, list_of_cp)
		milestone_closest_cp = neigh.predict(gmm.means_)

		assert len(milestone_closest_cp) == 3

		for cp in milestone_closest_cp:
			demonstration = self.map_cp2demonstrations[cp]
			surgeme = self.map_frm2surgeme[demonstration][self.map_cp2frm[cp]]
			frm = utils.get_frame_fig_name(self.map_cp2frm[cp])

			from_path = constants.PATH_TO_SUTURING_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_capture2/" + frm

			to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/milestones/" + self.map_cp2milestone[cp] + "_" + str(surgeme) + "_" + demonstration + "_" + frm

			utils.sys_copy(from_path, to_path)		

	def check_pruning_condition(self, list_of_cp):
		return len(list_of_cp) <= 5

	def copy_frames(self, demonstration, frm, l1_cluster, l2_cluster, surgeme):

		from_path = constants.PATH_TO_SUTURING_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_capture2/" + utils.get_frame_fig_name(frm)

		to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/" + l1_cluster + "/" + l2_cluster + "_" + str(surgeme) + "_" + demonstration + "_" + utils.get_frame_fig_name(frm)

		utils.sys_copy(from_path, to_path)

	def cluster_evaluation(self):

		all_surgemes = []
		for cp in self.list_of_cp:
			all_surgemes.append(self.map_frm2surgeme[self.map_cp2demonstrations[cp]][self.map_cp2frm[cp]])
		all_surgemes = set(all_surgemes)

		# Initialize data structures
		table = {}
		for L1_cluster in self.map_level1_cp.keys():
			new_dict = {}
			for surgeme in all_surgemes:
				new_dict[surgeme] = 0
			table[L1_cluster] = new_dict

		surgeme_count = {}
		for surgeme in all_surgemes:
			surgeme_count[surgeme] = 0

		for L1_cluster in self.map_level1_cp.keys():
			list_of_cp = self.map_level1_cp[L1_cluster]
			for cp in list_of_cp:
				surgeme = self.map_frm2surgeme[self.map_cp2demonstrations[cp]][self.map_cp2frm[cp]]
				surgeme_count[surgeme] += 1

				curr_dict = table[L1_cluster]
				curr_dict[surgeme] += 1
				table[L1_cluster] = curr_dict

		final_clusters = list(set(self.map_level1_cp.keys()) - set(self.pruned_L1_clusters))
		print self.pruned_L1_clusters

		print surgeme_count

		result = "   "
		for surgeme in all_surgemes:
			result = result + str(surgeme) + "     "
		print result
		result = ""
		for L1_cluster in final_clusters:
			result = result + L1_cluster + "   "
			for surgeme in all_surgemes:
				# result += str(float("{0:.2f}".format(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])))) + "   "
				result += str(round(Decimal(table[L1_cluster][surgeme] / float(surgeme_count[surgeme])), 2)) + "   "
			result += '\n'

		print result

	def do_everything(self):

		self.construct_features()

		self.generate_transition_features()

		self.generate_change_points()

		self.cluster_changepoints_level1()

		self.cluster_changepoints_level2()

		self.cluster_evaluation()

if __name__ == "__main__":
	mc = MilestonesClustering()
	mc.do_everything()