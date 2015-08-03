#!/usr/bin/env python

import IPython
import pickle
import numpy as np
import random
import os

import constants_sk as constants
import parser
import utils

from sklearn import (mixture, cluster, preprocessing, neighbors, random_projection)

class MilestonesClustering():
	def __init__(self):
		# self.list_of_demonstrations = parser.generate_list_of_videos(constants.PATH_TO_SUTURING_DATA + constants.CONFIG_FILE)
		self.list_of_demonstrations = ['Suturing_E001', 'Suturing_E002','Suturing_E003', 'Suturing_E004', 'Suturing_E005']
		self.data_X = {}
		self.data_N = {}

		self.change_pts = None
		self.change_pts_Z = None
		self.change_pts_W = None

		self.map_cp2frm = {}
		self.map_cp2demonstrations = {}
		self.map_cp_level1 = {}
		self.map_cp_level2 = {}
		self.map_level1_cp = {}
		self.map_level2_cp = {}
		self.map_cp_milestone = {}
		self.l2_cluster_matrices = {}
		self.map_frm2surgeme = parser.get_all_frame2surgeme_maps(self.list_of_demonstrations)
		self.trial = utils.hashcode()

		self.surgeme_level1_matrix = {}

	def get_kinematic_features(self, demonstration, sampling_rate = 1):
		return parser.parse_kinematics(constants.PATH_TO_SUTURING_KINEMATICS, constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_capture2.p",
			demonstration + ".txt", sampling_rate = sampling_rate)

	def get_visual_features(self, demonstration, PC = 100):
		layer = 'pool5'

		Z = pickle.load(open(constants.PATH_TO_SUTURING_DATA + constants.ALEXNET_FEATURES_FOLDER + layer
			+ "_alexnet_" + demonstration + "_capture2.p", "rb"))

		transformer = random_projection.GaussianRandomProjection(n_components=PC)	
		#return utils.pca(Z.astype(np.float), PC)
		projZ = transformer.fit_transform(Z)
		rotInvZ = preprocessing.normalize(projZ,norm='l2',axis=1)
		return rotInvZ

	def construct_features(self):
		for demonstration in self.list_of_demonstrations:
			print "Parsing Kinematics " + demonstration
			W = self.get_kinematic_features(demonstration)
			# scaler = preprocessing.StandardScaler().fit(W)
			# W = scaler.transform(W)

			print "Parsing visual features " + demonstration
			Z = self.get_visual_features(demonstration, PC = 1000)

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
			# Fit a Dirichlet process mixture of Gaussians using five components
			gmm = mixture.GMM(n_components = 10, covariance_type='diag')
			gmm.fit(N)
			Y = gmm.predict(N)

			map_cp2frm = {}
			for i in range(len(Y) - 1):
				print Y[i]
				if Y[i] != Y[i + 1]:
					change_pt = N[i][N.shape[1]/2:]
					self.append_cp_array(change_pt)
					self.map_cp2frm[cp_index] = i + 1
					self.map_cp2demonstrations[cp_index] = demonstration
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

		#sc = cluster.SpectralClustering(n_clusters=6, gamma=0.25, n_neighbors=10)  
		#sc.fit(self.change_pts_Z)
		gmm = mixture.GMM(n_components = 6, covariance_type='full')
		gmm.fit(self.change_pts_Z)
		Y = gmm.predict(self.change_pts_Z)
		#Y = sc.fit_predict(self.change_pts_Z)

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

			n_components = 5
			gmm = mixture.GMM(n_components = n_components, covariance_type='full')

			try:
				gmm.fit(matrix)

			except ValueError as e:

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

				if frm not in self.map_frm2surgeme[demonstration]:
					print "Frame: ", frm, "not found"
					continue

				surgeme = self.map_frm2surgeme[demonstration][frm]
	
				if (key,surgeme) not in self.surgeme_level1_matrix:
					self.surgeme_level1_matrix[(key, surgeme)] = 0

				self.surgeme_level1_matrix[(key, surgeme)] = self.surgeme_level1_matrix[(key, surgeme)] + 1

				if (-1,surgeme) not in self.surgeme_level1_matrix:
                                        self.surgeme_level1_matrix[(-1,surgeme)] = 0

				self.surgeme_level1_matrix[(-1, surgeme)] = self.surgeme_level1_matrix[(-1,surgeme)] + 1

				self.map_cp_milestone[cp] = milestone

				print("%s             %3d         %s   %3d   %3d    %3d" % (l1_cluster, l2_cluster, demonstration, frm, cp, surgeme))
				file.write("%s             %3d         %s   %3d   %3d    %3d\n" % (l1_cluster, l2_cluster, demonstration, frm, cp, surgeme))

				self.copy_frames(demonstration, frm, str(l1_cluster), str(l2_cluster), surgeme)

			self.copy_milestone_frames(matrix, list_of_cp, gmm)

	def copy_milestone_frames(self, matrix, list_of_cp, gmm):
		neigh = neighbors.KNeighborsClassifier(n_neighbors = 1)
		neigh.fit(matrix, list_of_cp)
		milestone_closest_cp = neigh.predict(gmm.means_)

		assert len(milestone_closest_cp) == 5

		for cp in milestone_closest_cp:
			demonstration = self.map_cp2demonstrations[cp]
			
			frm = self.map_cp2frm[cp]
			if frm not in self.map_frm2surgeme[demonstration]:
				print "Frame: ", frm, "not found"
				continue

			surgeme = self.map_frm2surgeme[demonstration][self.map_cp2frm[cp]]
			frm = utils.get_frame_fig_name(self.map_cp2frm[cp])

			from_path = constants.PATH_TO_SUTURING_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_capture2/" + frm

			to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/milestones/" + self.map_cp_milestone[cp] + "_" + str(surgeme) + "_" + demonstration + "_" + frm

			utils.sys_copy(from_path, to_path)		

	def check_pruning_condition(self, list_of_cp):
		return len(list_of_cp) <= 3

	def printSurgemeLevel1Matrix(self):
		result1 = []
		for key in self.surgeme_level1_matrix:
			if key[0] != -1:
				result1.append((key, (self.surgeme_level1_matrix[key] + 0.0)/self.surgeme_level1_matrix[(-1,key[1])]))
		result1.sort()

		print "Printing Surgeme-Cluster Matrix"
		for r in result1:
			print r

	def copy_frames(self, demonstration, frm, l1_cluster, l2_cluster, surgeme):

		from_path = constants.PATH_TO_SUTURING_DATA + constants.NEW_FRAMES_FOLDER + demonstration +"_capture2/" + utils.get_frame_fig_name(frm)

		to_path = constants.PATH_TO_CLUSTERING_RESULTS + self.trial + "/" + l1_cluster + "/" + l2_cluster + "_" + str(surgeme) + "_" + demonstration + "_" + utils.get_frame_fig_name(frm)

		utils.sys_copy(from_path, to_path)

	def do_everything(self):

		self.construct_features()

		self.generate_transition_features()

		self.generate_change_points()

		self.cluster_changepoints_level1()

		self.cluster_changepoints_level2()

		self.printSurgemeLevel1Matrix()

if __name__ == "__main__":
	mc = MilestonesClustering()
	mc.do_everything()
