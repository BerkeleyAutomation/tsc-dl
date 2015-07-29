#!/usr/bin/env python

import IPython
import pickle
import numpy as np

import constants
import parser
import utils

from sklearn import (mixture, preprocessing)

class MilestonesClustering():
	def __init__(self):
		# self.list_of_demonstrations = parser.generate_list_of_videos(constants.PATH_TO_SUTURING_DATA + constants.CONFIG_FILE)
		self.list_of_demonstrations = ['Suturing_E001', 'Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005']
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
		self.level2_cluster_matrices = {}
		pass

	def get_kinematic_features(self, demonstration, sampling_rate = 1):
		return parser.parse_kinematics(constants.PATH_TO_SUTURING_KINEMATICS, constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_capture2.p",
			demonstration + ".txt", sampling_rate = sampling_rate)

	def get_visual_features(self, demonstration, PC = 100):
		# layer = 'conv4'
		# data = pickle.load(open(constants.PATH_TO_SUTURING_DATA + constants.ALEXNET_FEATURES_FOLDER + "100samplingrate/"
		# 	+ "alexnet_" + demonstration + "_capture2.p", "rb"))
		# Z = data[0][layer]
		# return utils.pca(Z, PC)
		layer = 'pool5'
		Z = pickle.load(open(constants.PATH_TO_SUTURING_DATA + constants.ALEXNET_FEATURES_FOLDER + layer
			+ "_alexnet_" + demonstration + "_capture2.p", "rb"))
		return utils.pca(Z, PC)

	def construct_features(self):
		# sampling_rate = 100
		sampling_rate = 1
		for demonstration in self.list_of_demonstrations:
			print "Parsing Kinematics " + demonstration
			W = self.get_kinematic_features(demonstration, sampling_rate = sampling_rate).astype(np.float)
			print "Parsing visual features " + demonstration
			Z = self.get_visual_features(demonstration, PC = 100).astype(np.float)
			# Z = preprocessing.normalize(Z, norm = "l2")
			# W = preprocessing.normalize(W, norm = "l2")
			# min_max_scaler = preprocessing.MinMaxScaler()
			# Z = min_max_scaler.fit_transform(Z)
			# W = min_max_scaler.fit_transform(W)
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
		print "XXXX1"
		# IPython.embed()
		for i in range(len(Y)):
			label = constants.alphabet_map[Y[i] + 1]
			self.map_cp_level1[i] = label
			utils.dict_insert_list(label, i, self.map_level1_cp)
		print "XXXX2"
		# IPython.embed()	
		self.generate_level2_cluster_matrices()

	def generate_level2_cluster_matrices(self):
		for key in self.map_level1_cp.keys():
			list_of_cp = self.map_level1_cp[key]
			matrix = None
			for cp_index in list_of_cp:
				cp = utils.reshape(self.change_pts_W[cp_index])
				if matrix is None:
					matrix = cp
				else:
					matrix = np.concatenate((matrix, cp), axis = 0)
			self.level2_cluster_matrices[key] = matrix
		print "XXXX3"
		# IPython.embed()

	def cluster_changepoints_level2(self):
		print "Level2 : Clustering changepoints in W(t)"
		for key in self.map_level1_cp.keys():
			matrix = self.level2_cluster_matrices[key]
			list_of_cp = self.map_level1_cp[key]
			n_components = min(len(list_of_cp), 5)
			gmm = mixture.GMM(n_components = n_components, covariance_type='full')
			print "XXXX4"
			# IPython.embed()
			gmm.fit(matrix)
			Y = gmm.predict(matrix)
			for i in range(len(Y)):
				cp = list_of_cp[i]
				milestone = key+str(Y[i])
				self.map_cp_milestone[cp] = milestone
				print("CP: %3d Milestone: %s Frm number: %3d Demonstration: %s" %
					(cp, milestone, self.map_cp2frm[cp], self.map_cp2demonstrations[cp]))

	def do_everything(self):

		self.construct_features()

		self.generate_transition_features()

		self.generate_change_points()

		self.cluster_changepoints_level1()

		self.cluster_changepoints_level2()

if __name__ == "__main__":
	mc = MilestonesClustering()
	mc.do_everything()