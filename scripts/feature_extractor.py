#!/usr/bin/env python
import IPython
import pickle
import numpy as np

import constants

from forward_pass import CNNFeatureExtractor

# layers_of_interest = {"VGG": ['conv5_1', 'conv5_3'], "VGG_SOS": ['conv5_1', 'conv5_3'], "AlexNet": ["conv3", "conv4", "pool5"]}

layers_of_interest = {"VGG": ['conv5_3'], "VGG_SOS": ['conv5_3'], "AlexNet": ["conv4"]}

features_folder = {"VGG": constants.VGG_FEATURES_FOLDER, "VGG_SOS": constants.VGG_FEATURES_FOLDER, "AlexNet": constants.ALEXNET_FEATURES_FOLDER}

def forward_pass_entire_dataset(list_of_demonstrations, net_name, camera):
	"""
	Function performs forward pass of the frames corresponding to the demonstration
	through specified 2D CNN.
	Input: List of demonstrations, the Net (VGG, AlexNet) and camera (capture1, caputure2)
	"""
	net = CNNFeatureExtractor(net_name)
	
	total = len(list_of_demonstrations) 
	i = 1
	for video in list_of_demonstrations:
		print "-------------------- " + str(i) +"--out-of-"+ str(total)+"-----------------"
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + video + "_" + camera + ".p"
		PATH_TO_DATA = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + video + "_" + camera + "/"
		get_cnn_features_pickle_dump(net, video + "_" + camera, PATH_TO_DATA, PATH_TO_ANNOTATION, net_name)
		i += 1

def get_cnn_features_pickle_dump(net, fname, PATH_TO_DATA, annotations, net_name):
	"""
	Extracts CNN features for frames located at PATH_TO_DATA, saves them as pickle file with file name fname.
	"""

	list_of_layers = layers_of_interest[net_name]
	Z = net.forward_pass(PATH_TO_DATA, annotations, list_of_layers = list_of_layers, sampling_rate = 1, no_plot_mode = True)
	for key in Z.keys():
		pickle.dump(Z[key], open(constants.PATH_TO_DATA + features_folder[net_name] + key + "_" + net_name + "_" + fname + ".p", "wb"))

if __name__ == "__main__":
	# list_of_demonstrations = ['Suturing_D005', 'Suturing_C001', 'Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
	# 'Suturing_F001', 'Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']

	# list_of_demonstrations = ['Suturing_H001', 'Suturing_G003']

	# list_of_demonstrations = ['Suturing_G002', 'Suturing_G004', 'Suturing_G005',
	# 'Suturing_H003', 'Suturing_H004', 'Suturing_H005',
	# 'Suturing_I001', 'Suturing_I002', 'Suturing_I003', 'Suturing_I004', 'Suturing_I005']

	# list_of_demonstrations = ['Suturing_G003', 'Suturing_G004', 'Suturing_G005', 'Suturing_I004', 'Suturing_I005']

	# list_of_demonstrations = ['Suturing_I001', 'Suturing_I002', 'Suturing_I003', 'Suturing_I004', 'Suturing_I005']

	# list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
	# "Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

	# list_of_demonstrations = ["1001_01", "1001_02", "1001_03", "1001_04", "1001_05"]

	# list_of_demonstrations = ["plane_5",
	# 	"plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

	# list_of_demonstrations = ["people_0", "people_1", "people_2", "people_3", "people_4", "people_5", "people_6"]

	# list_of_demonstrations = ["lego_2", "lego_3", "lego_4", "lego_5", "lego_6", "lego_7"]

	# list_of_demonstrations = ["people2_2", "people2_3", "people2_4", "people2_5", "people2_6", "people2_7"]

	list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004",
	"Needle_Passing_E005", "Needle_Passing_C001", "Needle_Passing_C002"]

	forward_pass_entire_dataset(list_of_demonstrations, "VGG", constants.CAMERA)
	# forward_pass_entire_dataset(list_of_demonstrations, "AlexNet", constants.CAMERA)
