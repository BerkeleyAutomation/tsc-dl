#!/usr/bin/env python

import IPython
import pickle
import numpy as np
import random
import argparse

import constants
import parser
import utils

from sklearn.decomposition import PCA, IncrementalPCA

# Table of features:

# 1 - HOG-SIFT
# 2 - conv4 AlexNet
# 3 - conv3 AlexNet
# 4 - pool5 AlexNet
# 5 - conv X VGG
# 6 - conv Y VGG
# 7 - VGG conv5-1 + LCD + VLAD-k (each batch is 30 frames)
# 8 - Hypercolumns (AlexNet conv 3,4,5)
# 9 - Background Subtraction (conv4 AlexNet)
# 10 - Background Subtraction (conv X VGG)

PATH_TO_FEATURES = constants.PATH_TO_SUTURING_DATA + "features/"

def load_cnn_features(demonstration, layer, folder):
	Z = pickle.load(open(constants.PATH_TO_SUTURING_DATA + folder + layer
		+ "_alexnet_" + demonstration + "_capture2.p", "rb"))
	return Z.astype(np.float)

def get_kinematic_features(demonstration):
	return parser.parse_kinematics(constants.PATH_TO_SUTURING_KINEMATICS, constants.PATH_TO_SUTURING_DATA
		+ constants.ANNOTATIONS_FOLDER + demonstration + "_capture2.p", demonstration + ".txt")

def featurize(DEBUG = False):
	if DEBUG:
		list_of_demonstrations = ['Suturing_E005',]
	else:
		list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
		'Suturing_D001', 'Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005']

	# Parse Kinematic Features
	print "Parsing Kinematic Features"
	kinematics = {}
	for demonstration in list_of_demonstrations:
		W = get_kinematic_features(demonstration)
		kinematics[demonstration] = W

	featurize_4(list_of_demonstrations, kinematics)

	# featurize_7(list_of_demonstrations, kinematics)

	pass

# Featurize - AlexNet conv4
def featurize_2(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv4", constants.ALEXNET_FEATURES_FOLDER, 4)

# Featurize - AlexNet conv3
def featurize_3(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv3", constants.ALEXNET_FEATURES_FOLDER, 4)

# Featurize - AlexNet pool5
def featurize_4(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5", constants.ALEXNET_FEATURES_FOLDER, 4)

# Featurize - VGG conv5-1
def featurize_5(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5", constants.ALEXNET_FEATURES_FOLDER, 4)

# Featurize - VGG conv5-3
def featurize_6(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5", constants.ALEXNET_FEATURES_FOLDER, 4)

# Featurize - VGG + LCD + VLAD
def featurize_7(list_of_demonstrations, kinematics):
	for demonstration in list_of_demonstrations:
		W = kinematics[demonstration]
		Z = load_cnn_features(demonstration, "conv5-1", constants.VGG_FEATURES_FOLDER)

def featurize_cnn_features(list_of_demonstrations, kinematics, layer, folder, feature_index):

	data_X_PCA = {}
	data_X_CCA = {}

	big_Z = None

	# Initialization of data structs
	demonstration_size = {}
	init_demonstration = list_of_demonstrations[0]
	big_Z = load_cnn_features(init_demonstration, layer, folder)
	demonstration_size[init_demonstration] = big_Z.shape[0]
	
	for demonstration in list_of_demonstrations[1:]:
		print "Loading Visual Features for ", demonstration
		Z = load_cnn_features(demonstration, sampling_rate = sr)
		big_Z = np.concatenate((big_Z, Z), axis = 0)
		demonstration_size[demonstration] = Z.shape[0]


	big_Z_pca = utils.pca_incremental(big_Z, PC = 100)
	start = 0
	end = 0

	import matlab
	import matlab.engine as mateng

	eng = mateng.start_matlab()

	for demonstration in list_of_demonstrations:

		# ------------- PCA ------------- 
		W = kinematics[demonstration]

		size = demonstration_size[demonstration]
		end = start + size
		Z = big_Z_pca[start:end]
		start += size

		X = np.concatenate((W, Z), axis = 1)
		data_X_PCA[demonstration] = X

		# ------------- CCA ------------- 
		W_mat = matlab.double(W.tolist())
		Z_mat = matlab.double(Z.tolist())

		[A, B, r, U, V, stats] = eng.canoncorr(W_mat, Z_mat, nargout = 6)

		Z = np.array(V)
		X = np.concatenate((W, Z), axis = 1)
		data_X_CCA[demonstration] = X

	pickle.dump(data_X_PCA, open(PATH_TO_FEATURES + feature_index + "_PCA.p", "wb"))
	pickle.dump(data_X_CCA, open(PATH_TO_FEATURES + feature_index + "_CCA.p", "wb"))


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	args = argparser.parse_args()

	DEBUG = False
	if args.debug == 'y':
		DEBUG = True
	featurize(DEBUG)