#!/usr/bin/env python
import IPython
import pickle
import numpy as np

import constants

from forward_pass import CNNFeatureExtractor

layers_of_interest = {"VGG": ['conv5_1', 'conv5_3'], "VGG_SOS": ['conv5_1', 'conv5_3'], "AlexNet": ["conv3", "conv4", "pool5"]}

features_folder = {"VGG": constants.VGG_FEATURES_FOLDER, "VGG_SOS": constants.VGG_FEATURES_FOLDER, "AlexNet": constants.ALEXNET_FEATURES_FOLDER}

def forward_pass_entire_dataset(list_of_demonstrations, net_name, camera):
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
	# Note - Only storing features for conv1-5 and pool5

	list_of_layers = layers_of_interest[net_name]
	Z = net.forward_pass(PATH_TO_DATA, annotations, list_of_layers = list_of_layers, sampling_rate = 1, no_plot_mode = True)
	for key in Z.keys():
		pickle.dump(Z[key], open(constants.PATH_TO_DATA + features_folder[net_name] + key + "_" + net_name + "_" + fname + ".p", "wb"))

if __name__ == "__main__":
	list_of_demonstrations = ['Suturing_D005', 'Suturing_C001', 'Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
	'Suturing_F001', 'Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']

	# list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
	# "Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

	# list_of_demonstrations = ["1001_01", "1001_02", "1001_03", "1001_04", "1001_05"]

	# list_of_demonstrations = ["plane_3", "plane_4", "plane_5",
	# 	"plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

	forward_pass_entire_dataset(list_of_demonstrations, "AlexNet", constants.CAMERA)
	# forward_pass_entire_dataset(list_of_demonstrations, "VGG", constants.CAMERA)