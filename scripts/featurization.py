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
import sift

# Table of features:

# 1 - SIFT
# 2 - conv4 AlexNet
# 3 - conv3 AlexNet
# 4 - pool5 AlexNet
# 5 - conv5_3 VGG
# 6 - conv5_1 VGG
# 7 - VGG conv5_1 + LCD + VLAD-k (each batch is 30 frames)
# 8 - AlexNet conv_4 + LCD + VLAD-k

PATH_TO_FEATURES = constants.PATH_TO_DATA + constants.PROC_FEATURES_FOLDER

def load_cnn_features(demonstration, layer, folder, net):
	Z = pickle.load(open(constants.PATH_TO_DATA + folder + layer
		+ "_" + net + "_" + demonstration + "_" + constants.CAMERA +".p", "rb"))
	return Z.astype(np.float)

def main(DEBUG = False):
	if DEBUG:
		list_of_demonstrations = ['Suturing_E005',]
	else:

		# list_of_demonstrations = ["plane_3", "plane_4", "plane_5",
		# 	"plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

		# list_of_demonstrations = ["plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

		# list_of_demonstrations = ["011_01", "011_02", "011_03", "011_04", "011_05"]

		list_of_demonstrations = ["010_01", "010_02", "010_03", "010_04", "010_05"]

		# list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
		# "Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

		# list_of_demonstrations = ["people_0", "people_1", "people_2", "people_3", "people_4", "people_5", "people_6"]

		# list_of_demonstrations = ['Suturing_G002', 'Suturing_G004', 'Suturing_G005',
		# 'Suturing_H003', 'Suturing_H004', 'Suturing_H005',
		# 'Suturing_I001', 'Suturing_I002', 'Suturing_I003', 'Suturing_I004', 'Suturing_I005']

		# list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
		# 'Suturing_D001','Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
		# 'Suturing_C001','Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
		# 'Suturing_F001','Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005',
		# 'Suturing_G002', 'Suturing_G004', 'Suturing_G005',
		# 'Suturing_H003', 'Suturing_H004', 'Suturing_H005',
		# 'Suturing_I001', 'Suturing_I002', 'Suturing_I003', 'Suturing_I004', 'Suturing_I005',
		# 'Suturing_B001', 'Suturing_B002', 'Suturing_B003', 'Suturing_B004', 'Suturing_B005']

		# list_of_demonstrations = ["Needle_Passing_B001", "Needle_Passing_B002", "Needle_Passing_B003", "Needle_Passing_B004",
		# "Needle_Passing_C001","Needle_Passing_C002", "Needle_Passing_C003", "Needle_Passing_C004", "Needle_Passing_C005",
		# "Needle_Passing_D001","Needle_Passing_D002", "Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005",
		# "Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
		# "Needle_Passing_F001", "Needle_Passing_F003", "Needle_Passing_F004",
		# "Needle_Passing_H002", "Needle_Passing_H004", "Needle_Passing_H005",
		# "Needle_Passing_I002", "Needle_Passing_I003", "Needle_Passing_I004","Needle_Passing_I005"]

		# Experts + Intermediates (Needle Passing)
		# list_of_demonstrations = ["Needle_Passing_C001","Needle_Passing_C002", "Needle_Passing_C003", "Needle_Passing_C004", "Needle_Passing_C005",
		# "Needle_Passing_D001","Needle_Passing_D002", "Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005",
		# "Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
		# "Needle_Passing_F001", "Needle_Passing_F003", "Needle_Passing_F004"]

		# list_of_demonstrations = ["lego_2", "lego_3", "lego_4", "lego_5", "lego_6", "lego_7"]

		# list_of_demonstrations = ["people2_2", "people2_3", "people2_4", "people2_5", "people2_6", "people2_7"]

		list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005']

	# Parse Kinematic Features
	print "Parsing Kinematic Features"
	kinematics = {}
	for demonstration in list_of_demonstrations:
		W = utils.get_kinematic_features(demonstration)
		kinematics[demonstration] = W
	sr = constants.SR
	print "Sampling rate:",sr
	featurize_sift(list_of_demonstrations, kinematics, sr)
	# featurize_1(list_of_demonstrations, kinematics, sr)
	# featurize_2(list_of_demonstrations, kinematics, sr)
	# featurize_3(list_of_demonstrations, kinematics, sr)
	# featurize_4(list_of_demonstrations, kinematics, sr)
	# featurize_5(list_of_demonstrations, kinematics, sr)
	# featurize_6(list_of_demonstrations, kinematics, sr)
	# featurize_7(list_of_demonstrations, kinematics, sr)
	# featurize_8(list_of_demonstrations, kinematics, sr)
	# featurize_visual(list_of_demonstrations, sr)
	featurize_PCA(list_of_demonstrations, kinematics, sr)
	pass



def featurize_sift(list_of_demonstrations, kinematics, sr):
	"""
	Extracts SIFT features for all frames in list_of_demonstrations.
	"""

	data_X_xy = {}
	data_X_x = {}

	for demonstration in list_of_demonstrations:
		print "SIFT for ", demonstration
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"

		Z = []
		start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)
		for frm in range(start, end + 1):
			PATH_TO_IMAGE = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + constants.CAMERA + "/"
			Z.append(sift.run_surf_frame(utils.get_full_image_path(PATH_TO_IMAGE, frm)))

		Z = np.array(Z)
		Z = Z.reshape(Z.shape[0],1)

		W = kinematics[demonstration]
		W_onlyx = utils.only_X(W)

		X = np.concatenate((W, Z), axis = 1)
		X_onlyx = np.concatenate((W_onlyx, Z), axis = 1)

		data_X_xy[demonstration] = X
		data_X_x[demonstration] = X_onlyx

	pickle.dump(data_X_xy, open(PATH_TO_FEATURES + "SIFT_xy.p", "wb"))
	pickle.dump(data_X_x, open(PATH_TO_FEATURES + "SIFT_x.p", "wb"))

# Featurize - SIFT
def featurize_1(list_of_demonstrations, kinematics, sr):
	"""
	Given .p (pickle) files of sift features, this function concatenates
	Z with W vectors to produce X vector, dumped as pickle file.
	"""
	print "FEATURIZATION 1"

	data_X_1 = {}
	data_X_2 = {}
	for demonstration in list_of_demonstrations:
		print "SIFT for ", demonstration
		start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
						+ demonstration + "_" + constants.CAMERA +".p")

		W = kinematics[demonstration]
		W_sampled = utils.sample_matrix(W, sampling_rate = sr)


		PATH_TO_SIFT = constants.PATH_TO_DATA + "sift_FCED/SIFT_"+ demonstration
		Z = pickle.load(open(PATH_TO_SIFT + "_1.p", "rb"))
		Z = Z[start:end + 1]
		Z_sampled_1 = utils.sample_matrix(Z, sampling_rate = sr)

		Z = pickle.load(open(PATH_TO_SIFT + "_2.p", "rb"))
		Z = Z[start:end + 1]
		Z_sampled_2 = utils.sample_matrix(Z, sampling_rate = sr)

		assert Z_sampled_1.shape[0] == W_sampled.shape[0]
		assert Z_sampled_2.shape[0] == W_sampled.shape[0]

		data_X_1[demonstration] = np.concatenate((W_sampled, Z_sampled_1), axis = 1)
		data_X_2[demonstration] = np.concatenate((W_sampled, Z_sampled_2), axis = 1)

	pickle.dump(data_X_1, open(PATH_TO_FEATURES + "SIFT_1.p", "wb"))
	pickle.dump(data_X_2, open(PATH_TO_FEATURES + "SIFT_2.p", "wb"))

# Featurize - AlexNet conv4
def featurize_2(list_of_demonstrations, kinematics, sr):
	print "FEATURIZATION 2"
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv4",
		constants.ALEXNET_FEATURES_FOLDER, 2, "AlexNet", sr)

# Featurize - AlexNet conv3
def featurize_3(list_of_demonstrations, kinematics, sr):
	print "FEATURIZATION 3"
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv3",
		constants.ALEXNET_FEATURES_FOLDER, 3, "AlexNet", sr)

# Featurize - AlexNet pool5
def featurize_4(list_of_demonstrations, kinematics, sr):
	print "FEATURIZATION 4"
	featurize_cnn_features(list_of_demonstrations, kinematics, "pool5",
		constants.ALEXNET_FEATURES_FOLDER, 4, "AlexNet", sr)

# Featurize - VGG conv5_3
def featurize_5(list_of_demonstrations, kinematics, sr):
	print "FEATURIZATION 5"
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv5_3",
		constants.VGG_FEATURES_FOLDER, 5, "VGG", sr)

# Featurize - VGG conv5_1
def featurize_6(list_of_demonstrations, kinematics, sr):
	print "FEATURIZATION 6"
	featurize_cnn_features(list_of_demonstrations, kinematics, "conv5_1",
		constants.VGG_FEATURES_FOLDER, 6, "VGG", sr)

# Featurize - VGG conv5_3 + LCD + VLAD
def featurize_7(list_of_demonstrations, kinematics, sr, config = [True, True, True]):
	print "FEATURIZATION 7"
	layer = "conv5_3"
	net_name = "VGG"
	conv_dimensions = constants.caffe_conv_dimensions[layer]
	folder = constants.VGG_FEATURES_FOLDER
	FPS = 30
	# Batch size approximately captures 1 second of activity
	batch_size = int(FPS / float(sr))
	fname = "7"
	featurize_LCD_VLAD(list_of_demonstrations, kinematics, layer, net_name, folder, conv_dimensions, batch_size, fname, config = [True, True, True])

# Featurize - AlexNet conv4 + LCD + VLAD
def featurize_8(list_of_demonstrations, kinematics, sr, config = [True, True, True]):
	print "FEATURIZATION 8"
	layer = "conv4"
	net_name = "alexnet"
	conv_dimensions = constants.caffe_conv_dimensions[layer]
	folder = constants.ALEXNET_FEATURES_FOLDER
	FPS = 30
	# Batch size approximately captures 1 second of activity
	batch_size = int(FPS / float(sr))
	fname = "8"
	featurize_LCD_VLAD(list_of_demonstrations, kinematics, layer, net_name, folder, conv_dimensions, batch_size, fname, config = [True, True, True])

# Featurize - VGG conv5_3 with varing PCA dimensions
def featurize_PCA(list_of_demonstrations, kinematics, sr):
	print "FEATURIZATION 5"
	list_of_PC = [500, 2000]
	for PC in list_of_PC:
		featurize_cnn_features(list_of_demonstrations, kinematics, "conv5_3",
			constants.VGG_FEATURES_FOLDER, PC,"VGG", sr, C = PC)

def featurize_LCD_VLAD(list_of_demonstrations, kinematics, layer, net_name, folder, dimensions, batch_size, fname, config = [True, True, True]):
	M = dimensions[0]
	a = dimensions[1]

	print "Featurizing LCD + VLAD: ", layer, net_name, folder, M, a, batch_size

	BATCH_SIZE = batch_size

	if constants.SIMULATION:
		BATCH_SIZE = 5

	data_X_PCA = {}
	data_X_CCA = {}
	data_X_GRP = {}

	size_sampled_matrices = [utils.sample_matrix(kinematics[demo], sampling_rate = BATCH_SIZE).shape[0] for demo in list_of_demonstrations]
	PC = min(100, min(size_sampled_matrices))
	print "PC: ", PC

	for demonstration in list_of_demonstrations:
		print demonstration
		W = kinematics[demonstration]
		Z = load_cnn_features(demonstration, layer, folder, net_name)
		W_new = utils.sample_matrix(W, sampling_rate = BATCH_SIZE)

		Z_batch = None
		W_batch = None
		j = 1

		Z_new = None

		IPython.embed()

		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"
		start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)

		Z = []

		IPython.embed()

		for i in range(Z):

			vector_W = W[i]
			W_batch = utils.safe_concatenate(W_batch, vector_W)

			vector_Z = Z[i]
			vector_Z = vector_Z.reshape(M, a, a)
			vector_Z = lcd.LCD(vector_Z)
			Z_batch = utils.safe_concatenate(Z_batch, vector_Z)

			if (j == BATCH_SIZE):
				print "NEW BATCH", str(i)
				Z_batch_VLAD = encoding.encode_VLAD(Z_batch)
				Z_new = utils.safe_concatenate(Z_new, Z_batch_VLAD)

				# Re-initialize batch variables
				j = 0
				Z_batch = None
				W_batch = None

			j += 1

		# tail case
		if Z_batch is not None:
			print "TAIL CASE"
			print "NEW BATCH", str(i)
			Z_batch_VLAD = encoding.encode_VLAD(Z_batch)
			Z_new = utils.safe_concatenate(Z_new, Z_batch_VLAD)

		if config[0]:
			Z_new_pca = utils.pca_incremental(Z_new, PC = PC)
			print Z_new_pca.shape
			assert W_new.shape[0] == Z_new_pca.shape[0]
			X_PCA = np.concatenate((W_new, Z_new_pca), axis = 1)
			data_X_PCA[demonstration] = X_PCA

		if config[1]:
			Z_new_cca = utils.cca(W_new, Z_new)
			print Z_new_cca.shape
			assert W_new.shape[0] == Z_new_cca.shape[0]
			X_CCA = np.concatenate((W_new, Z_new_cca), axis = 1)
			data_X_CCA[demonstration] = X_CCA

		if config[2]:
			Z_new_grp = utils.grp(Z_new)
			print Z_new_grp.shape
			assert W_new.shape[0] == Z_new_grp.shape[0]
			X_GRP = np.concatenate((W_new, Z_new_grp), axis = 1)
			data_X_GRP[demonstration] = X_GRP

	if config[0]:
		pickle.dump(data_X_PCA, open(PATH_TO_FEATURES + fname + "_PCA" + ".p", "wb"))
	if config[1]:
		pickle.dump(data_X_CCA, open(PATH_TO_FEATURES + fname + "_CCA" + ".p", "wb"))
	if config[2]:
		pickle.dump(data_X_GRP, open(PATH_TO_FEATURES + fname + "_GRP" + ".p", "wb"))

def featurize_cnn_features(list_of_demonstrations, kinematics, layer, folder, feature_index, net, sr = 3, config = [True, False, False], C = 100):

	# For config params [x,y,z] refers to perform PCA, CCA and GRP respectively

	data_X_PCA = {}
	data_X_CCA = {}
	data_X_GRP = {}

	big_Z = None

	# Initialization
	demonstration_size = {}
	init_demonstration = list_of_demonstrations[0]
	big_Z = utils.sample_matrix(load_cnn_features(init_demonstration, layer, folder, net), sampling_rate = sr)
	demonstration_size[init_demonstration] = big_Z.shape[0]

	kinematics_sampled = {}
	kinematics_sampled[init_demonstration] = utils.sample_matrix(kinematics[init_demonstration], sampling_rate = sr)

	dimensions_of_PCA = demonstration_size[init_demonstration]

	for demonstration in list_of_demonstrations[1:]:
		print "Loading Visual Features for ", demonstration
		Z = load_cnn_features(demonstration, layer, folder, net)
		Z_sampled = utils.sample_matrix(Z, sampling_rate = sr)
		print "Sampled visual", Z.shape, Z_sampled.shape

		dimensions_of_PCA += Z_sampled.shape[0]

		big_Z = np.concatenate((big_Z, Z_sampled), axis = 0)
		demonstration_size[demonstration] = Z_sampled.shape[0]

		kinematics_sampled[demonstration] = utils.sample_matrix(kinematics[demonstration], sampling_rate = sr)
		print "Sampled kinematics", kinematics_sampled[demonstration].shape

		print "PCA # of rows: ", dimensions_of_PCA

	PC = min(C, min(demonstration_size.values()))
	print "XXXX - PC:", str(PC)

	# Quick check to see if kinematics and visual features are aligned
	for demonstration in list_of_demonstrations:
		print demonstration_size[demonstration], kinematics_sampled[demonstration].shape[0]
		assert demonstration_size[demonstration] == kinematics_sampled[demonstration].shape[0]

	if config[0]:
		big_Z_pca = utils.pca_incremental(big_Z, PC = PC)

	if config[2]:
		big_Z_grp = utils.grp(big_Z)

	start = 0
	end = 0

	import matlab
	import matlab.engine as mateng

	eng = mateng.start_matlab()

	for demonstration in list_of_demonstrations:

		W = kinematics_sampled[demonstration]

		size = demonstration_size[demonstration]
		end = start + size

		# ------------- PCA ------------- 
		if config[0]:
			Z = big_Z_pca[start:end]
			X = np.concatenate((W, Z), axis = 1)
			data_X_PCA[demonstration] = X

		# ------------- CCA ------------- 
		if config[1]:
			Z = big_Z[start:end]
			Z = utils.cca(W, Z)
			X = np.concatenate((W, Z), axis = 1)
			data_X_CCA[demonstration] = X

		# ------------- GRP -------------
		if config[2]:
			Z = big_Z_grp[start:end]
			X = np.concatenate((W, Z), axis = 1)
			data_X_GRP[demonstration] = X
		
		start += size

	if config[0]:
		pickle.dump(data_X_PCA, open(PATH_TO_FEATURES + str(feature_index) + "_PCA.p", "wb"))

	if config[1]:
		pickle.dump(data_X_CCA, open(PATH_TO_FEATURES + str(feature_index) + "_CCA.p", "wb"))

	if config[2]:
		pickle.dump(data_X_GRP, open(PATH_TO_FEATURES + str(feature_index) + "_GRP.p", "wb"))

def featurize_visual(list_of_demonstrations, sr):
	print "FEATURIZATION 2 - Visual"
	featurize_cnn_features_no_kinematics(list_of_demonstrations, "conv4",
		constants.ALEXNET_FEATURES_FOLDER, 2, "AlexNet", sr)
	print "FEATURIZATION 5 - Visual"
	featurize_cnn_features_no_kinematics(list_of_demonstrations, "conv5_3",
		constants.VGG_FEATURES_FOLDER, 5, "VGG", sr)

def featurize_cnn_features_no_kinematics(list_of_demonstrations, layer, folder, feature_index, net, sr = 3, config = [True, False, True]):

	# For config params [x,y,z] refers to perform PCA, CCA and GRP respectively

	data_X_PCA = {}
	data_X_GRP = {}

	big_Z = None

	# Initialization
	demonstration_size = {}
	init_demonstration = list_of_demonstrations[0]
	big_Z = utils.sample_matrix(load_cnn_features(init_demonstration, layer, folder, net), sampling_rate = sr)
	demonstration_size[init_demonstration] = big_Z.shape[0]

	for demonstration in list_of_demonstrations[1:]:
		print "Loading Visual Features for ", demonstration
		Z = load_cnn_features(demonstration, layer, folder, net)
		Z_sampled = utils.sample_matrix(Z, sampling_rate = sr)

		big_Z = np.concatenate((big_Z, Z_sampled), axis = 0)
		demonstration_size[demonstration] = Z_sampled.shape[0]

	PC = min(100, min(demonstration_size.values()))

	if config[0]:
		big_Z_pca = utils.pca_incremental(big_Z, PC = PC)

	if config[2]:
		big_Z_grp = utils.grp(big_Z)

	start = 0
	end = 0

	import matlab
	import matlab.engine as mateng

	eng = mateng.start_matlab()

	for demonstration in list_of_demonstrations:

		size = demonstration_size[demonstration]
		end = start + size

		# ------------- PCA ------------- 
		if config[0]:
			Z = big_Z_pca[start:end]
			data_X_PCA[demonstration] = Z

		# ------------- GRP -------------
		if config[2]:
			Z = big_Z_grp[start:end]
			data_X_GRP[demonstration] = Z
		
		start += size

	if config[0]:
		pickle.dump(data_X_PCA, open(PATH_TO_FEATURES + str(feature_index) + "_PCA.p", "wb"))

	if config[2]:
		pickle.dump(data_X_GRP, open(PATH_TO_FEATURES + str(feature_index) + "_GRP.p", "wb"))

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--debug", help = "Debug mode?[y/n]", default = 'n')
	args = argparser.parse_args()

	DEBUG = False
	if args.debug == 'y':
		DEBUG = True
	main(DEBUG)
