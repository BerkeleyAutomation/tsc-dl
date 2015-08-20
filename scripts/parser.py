#!/usr/bin/env python
import IPython
import pickle
import numpy as np

import constants

from forward_pass import CNNFeatureExtractor

layers_of_interest = {"VGG_SOS": ['conv5_1', 'conv5_3'], "AlexNet": ["conv3", "conv4", "pool5"]}

features_folder = {"VGG_SOS": constants.VGG_FEATURES_FOLDER, "AlexNet": constants.ALEXNET_FEATURES_FOLDER}

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

def generate_list_of_demonstrations(config_file_name, include_camera = False):
	list_of_demonstrations = []
	with open(config_file_name, "rb") as f:
		for line in f:
			params = line.split()
			if len(params) != 0:
				if include_camera:
					list_of_demonstrations.append(params[0] + "_capture1")
					list_of_demonstrations.append(params[0] + "_capture2")
				else:
					list_of_demonstrations.append(params[0])					
	return list_of_demonstrations

def frame2surgeme_map_demonstration(PATH_TO_TRANSCRIPTION, demonstration):
	map_frame2surgeme = {}
	with open(PATH_TO_TRANSCRIPTION + demonstration + ".txt", "rb") as f:
		for line in f:
			line = line.split()
			start = int(line[0])
			end = int(line[1])
			surgeme = int(constants.map_surgeme_label[line[2]])
			i = start
			while i <= end:
				map_frame2surgeme[i] = surgeme
				i += 1
	return map_frame2surgeme

def get_all_frame2surgeme_maps(list_of_demonstrations):
	"""
	For each demonstration in list_of_demonstrations, function returns a map:
	[frame number] -> segment label
	"""

	map_frame2surgeme = {}

	for demonstration in list_of_demonstrations:

		map_frame2surgeme[demonstration] = frame2surgeme_map_demonstration(constants.PATH_TO_DATA +
			constants.TRANSCRIPTIONS_FOLDER, demonstration)

	return map_frame2surgeme

def convert_transcription_to_annotation(PATH_TO_TRANSCRIPTION, PATH_TO_ANNOTATION, demonstration):
	"""
	Converts transcription.txt file to annotations.p file containing a dictionary of surgeme labels
	"""

	segments = {}
	with open(PATH_TO_TRANSCRIPTION + demonstration + ".txt", "rb") as f:
		for line in f:
			line = line.split()
			start = int(line[0])
			end = int(line[1])
			segment_index = int(constants.map_surgeme_label[line[2]])
			if segment_index not in segments:
				segments[segment_index] = [(start, end),]
			else:
				curr_list = segments[segment_index]
				curr_list.append((start, end))
				segments[segment_index] = curr_list

	pickle.dump(segments, open(PATH_TO_ANNOTATION + demonstration + "_capture1.p", "wb"))
	pickle.dump(segments, open(PATH_TO_ANNOTATION + demonstration + "_capture2.p", "wb"))

def parse_annotations():
	"""
	Note that left and right cameres have same transcriptions/annotations
	"""
	list_of_demonstrations = generate_list_of_demonstrations(constants.PATH_TO_DATA + constants.CONFIG_FILE)
	for video in list_of_demonstrations:
		convert_transcription_to_annotation(constants.PATH_TO_DATA + constants.TRANSCRIPTIONS_FOLDER,
			constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER, video)

def get_start_end_annotations(PATH_TO_ANNOTATION):
	"""
	Given the annotation (pickle) file, function returns the start and end
	frames of the demonstration.
	"""
	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	list_of_end_start_pts = []
	for key in segments:
		list_of_keys = segments[key]
		for elem in list_of_keys:
			list_of_end_start_pts.append(elem[0])
			list_of_end_start_pts.append(elem[1])
	start = min(list_of_end_start_pts)
	end = max(list_of_end_start_pts)
	return start, end

def get_annotation_segments(PATH_TO_ANNOTATION):
	segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
	list_of_end_start_pts = []
	for key in segments:
		list_of_keys = segments[key]
		for elem in list_of_keys:
			list_of_end_start_pts.append(elem)
	return list_of_end_start_pts

def parse_kinematics(PATH_TO_KINEMATICS_DATA, PATH_TO_ANNOTATION, fname, sampling_rate = 1):
	"""
	Takes in PATH to kinematics data (a txt file) and outputs a N x 38 matrix,
	where N is the number of frames. There are 38 dimensions in the kinematic data

	39-41  (3) : Slave left tooltip xyz
	42-50  (9) : Slave left tooltip R
	51-53  (3) : Slave left tooltip trans_vel x', y', z'   
	54-56  (3) : Slave left tooltip rot_vel
	57     (1) : Slave left gripper angle 
	58-76  (19): Slave right
	"""

	start, end = get_start_end_annotations(PATH_TO_ANNOTATION)

	X = None
	all_lines = open(PATH_TO_KINEMATICS_DATA + fname, "rb").readlines()
	i = start - 1
	if i < 0:
		i = 0 
	while i < end:
		traj = np.array(all_lines[i].split())
		slave = traj[38:]
		# slave_left = traj[38:57]
		# slave_right = traj[57:]
		if X is not None:
			X = np.concatenate((X, slave.reshape(1, slave.shape[0])), axis = 0)
		else:
			X = slave.reshape(1, slave.shape[0])		
		i += sampling_rate
	return X.astype(np.float)


if __name__ == "__main__":
	# list_of_demonstrations = ['Suturing_D005', 'Suturing_C001', 'Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
	# 'Suturing_F001', 'Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']

	list_of_demonstrations = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
	"Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]

	forward_pass_entire_dataset(list_of_demonstrations, "AlexNet", "capture1")