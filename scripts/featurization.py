#!/usr/bin/env python

import IPython
import pickle
import numpy as np
import random
import argparse

import constants
import parser
import utils
import lcd
import encoding

from sklearn.decomposition import PCA, IncrementalPCA

# Table of features:

# 1 - SIFT
# 2 - conv4 AlexNet
# 3 - conv3 AlexNet
# 4 - pool5 AlexNet
# 5 - conv5_1 VGG
# 6 - conv5_3 VGG
# 7 - VGG conv5_1 + LCD + VLAD-k (each batch is 30 frames)
# 8 - Hypercolumns (AlexNet conv 3,4,5)
# 9 - Background Subtraction (conv4 AlexNet)
# 10 - Background Subtraction (conv5_3 VGG)
# 11 - HOG
# 12 - SIFT v2

PATH_TO_FEATURES = constants.PATH_TO_SUTURING_DATA + constants.PROC_FEATURES_FOLDER

def load_cnn_features(demonstration, layer, folder, net):
	Z = pickle.load(open(constants.PATH_TO_SUTURING_DATA + folder + layer
		+ "_" + net + "_" + demonstration + "_capture2.p", "rb"))
	return Z.astype(np.float)

def get_kinematic_features(demonstration):
	return parser.parse_kinematics(constants.PATH_TO_SUTURING_KINEMATICS, constants.PATH_TO_SUTURING_DATA
		+ constants.ANNOTATIONS_FOLDER + demonstration + "_capture2.p", demonstration + ".txt")

def main(DEBUG = False):
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
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv4",
		constants.ALEXNET_FEATURES_FOLDER, 2, "alexnet")

# Featurize - AlexNet conv3
def featurize_3(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv3",
		constants.ALEXNET_FEATURES_FOLDER, 3, "alexnet")

# Featurize - AlexNet pool5
def featurize_4(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5",
		constants.ALEXNET_FEATURES_FOLDER, 4, "alexnet")

# Featurize - VGG conv5_1
def featurize_5(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5",
		constants.ALEXNET_FEATURES_FOLDER, 5, "vgg")

# Featurize - VGG conv5_3
def featurize_6(list_of_demonstrations, kinematics):
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5",
		constants.ALEXNET_FEATURES_FOLDER, 6, "vgg")

# Featurize - VGG + LCD + VLAD
def featurize_7(list_of_demonstrations, kinematics):
	a = 13 # Need to find the original values!!
	M = 9
	BATCH_SIZE = 15
	data_X = {}
	for demonstration in list_of_demonstrations:
		W = kinematics[demonstration]
		Z = load_cnn_features(demonstration, "conv5_3", constants.VGG_FEATURES_FOLDER, "vgg")
		W_new = utils.sample_matrix(sampling_rate = BATCH_SIZE)

		Z_new = None
		Z_batch = None
		j = 1

		for i in range(len(Z)):

			vector = Z[i]
			vector = vector.reshape(M, a, a)
			vector = lcd.LCD(vector)
			Z_batch = utils.safe_concatenate(Z_batch, vector)

			if (j == BATCH_SIZE):

				Z_batch_VLAD = encoding.VLAD(Z_batch)
				Z_new = utils.safe_concatenate(Z_new, Z_batch_VLAD)

				# Re-initialize batch variables
				j = 0
				Z_batch = None

			j += 1

		assert W_new.shape[0] == Z_new.shape[0]
		X = np.concatenate((W_new, Z_new), axis = 0)
		data_X[demonstration] = X

	pickle.dump(data_X, open(PATH_TO_FEATURES + str(7) + "_.p", "wb"))

def featurize_cnn_features(list_of_demonstrations, kinematics, layer, folder, feature_index, net):

	data_X_PCA = {}
	data_X_CCA = {}

	big_Z = None

	# Initialization of data structs
	demonstration_size = {}
	init_demonstration = list_of_demonstrations[0]
	big_Z = load_cnn_features(init_demonstration, layer, folder, net)
	demonstration_size[init_demonstration] = big_Z.shape[0]
	
	for demonstration in list_of_demonstrations[1:]:
		print "Loading Visual Features for ", demonstration
		Z = load_cnn_features(demonstration, layer, folder, net)
		big_Z = np.concatenate((big_Z, Z), axis = 0)
		demonstration_size[demonstration] = Z.shape[0]

	print "PCA....."
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

	pickle.dump(data_X_PCA, open(PATH_TO_FEATURES + str(feature_index) + "_PCA.p", "wb"))
	pickle.dump(data_X_CCA, open(PATH_TO_FEATURES + str(feature_index) + "_CCA.p", "wb"))


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	args = argparser.parse_args()

	DEBUG = False
	if args.debug == 'y':
		DEBUG = True
	main(DEBUG)