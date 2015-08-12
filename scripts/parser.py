#!/usr/bin/env python
import IPython
import pickle
import numpy as np

import constants

from forward_pass import CNNFeatureExtractor


def forward_pass_entire_dataset():
	net = CNNFeatureExtractor("VGG_SOS")
	# list_of_videos = generate_list_of_videos(constants.PATH_TO_SUTURING_DATA + constants.CONFIG_FILE)

	list_of_videos = ['Suturing_D001', 'Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
	'Suturing_C001', 'Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005']
	
	total = len(list_of_videos) 
	i = 1
	for video in list_of_videos:
		# print "-------------------- " + video +"_capture1 -----" + str(i) + "--out-of-"+ str(total)+"-----------------"
		# PATH_TO_ANNOTATION = constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER + video + "_capture1.p"
		# PATH_TO_DATA = constants.PATH_TO_SUTURING_DATA + constants.NEW_FRAMES_FOLDER + video + "_capture1/"
		# get_cnn_features_pickle_dump(net, video + "_capture1", PATH_TO_DATA, PATH_TO_ANNOTATION)
		# i += 1
		print "-------------------- " + video +"_capture2 -----" + str(i) +"--out-of-"+ str(total)+"-----------------"
		PATH_TO_ANNOTATION = constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER + video + "_capture2.p"
		PATH_TO_DATA = constants.PATH_TO_SUTURING_DATA + constants.NEW_FRAMES_FOLDER + video + "_capture2/"
		get_cnn_features_pickle_dump(net, video + "_capture2", PATH_TO_DATA, PATH_TO_ANNOTATION)
		i += 1

def get_cnn_features_pickle_dump(net, fname, PATH_TO_DATA, annotations):
	# Note - Only storing features for conv1-5 and pool5

	list_of_layers = ['conv5_1', 'conv5_3']
	Z = net.forward_pass(PATH_TO_DATA, annotations, list_of_layers = list_of_layers, sampling_rate = 1, no_plot_mode = True)
	for key in Z.keys():
		pickle.dump(Z[key], open(constants.PATH_TO_SUTURING_DATA + constants.VGG_FEATURES_FOLDER + key + "_vgg_" + fname + ".p", "wb"))

def generate_list_of_videos(config_file_name, include_camera = False):
	list_of_videos = []
	with open(config_file_name, "rb") as f:
		for line in f:
			params = line.split()
			if len(params) != 0:
				if include_camera:
					list_of_videos.append(params[0] + "_capture1")
					list_of_videos.append(params[0] + "_capture2")
				else:
					list_of_videos.append(params[0])					
	return list_of_videos

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

def get_all_frame2surgeme_maps(list_of_videos):
	map_frame2surgeme = {}

	for demonstration in list_of_videos:

		map_frame2surgeme[demonstration] = frame2surgeme_map_demonstration(constants.PATH_TO_SUTURING_DATA +
			constants.TRANSCRIPTIONS_FOLDER, demonstration)

	return map_frame2surgeme

def convert_transcription_to_annotation(PATH_TO_TRANSCRIPTION, PATH_TO_ANNOTATION, demonstration):
	segments = {}
	with open(PATH_TO_TRANSCRIPTION + demonstration + ".txt", "rb") as f:
		for line in f:
			print line
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
	list_of_videos = generate_list_of_videos(constants.PATH_TO_SUTURING_DATA + constants.CONFIG_FILE)
	# Note that left and right cameres have same transcriptions/annotations
	for video in list_of_videos:
		convert_transcription_to_annotation(constants.PATH_TO_SUTURING_DATA + constants.TRANSCRIPTIONS_FOLDER,
			constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER, video)

def get_start_end_annotations(PATH_TO_ANNOTATION):
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

def parse_kinematics(PATH_TO_KINEMATICS_DATA, PATH_TO_ANNOTATION, fname, sampling_rate = 1):
	# 39-41  (3) : Slave left tooltip xyz
	# 42-50  (9) : Slave left tooltip R
	# 51-53  (3) : Slave left tooltip trans_vel x', y', z'   
	# 54-56  (3) : Slave left tooltip rot_vel
	# 57     (1) : Slave left gripper angle 
	# 58-76  (19): Slave right

	# X = None
	# with open(PATH_TO_KINEMATICS_DATA + fname, "rb") as f:
	# 	i = 0
	# 	print "Inside parse_kinematics"
	# 	IPython.embed()
	# 	# for line in f:
	# 	# 	if i % sampling_rate == 0:
	# 	# 		traj = np.array(line.split())
	# 	# 		slave = traj[38:]
	# 	# 		slave_left = traj[38:57]
	# 	# 		slave_right = traj[57:]
	# 	# 		if X is not None:
	# 	# 			X = np.concatenate((X, slave.reshape(1, slave.shape[0])), axis = 0)
	# 	# 		else:
	# 	# 			X = slave.reshape(1, slave.shape[0])
	# 	# 	i += 1
	# return X

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
	# with open(PATH_TO_KINEMATICS_DATA + fname, "rb") as f:
	# 	i = 0
	# 	print "Inside parse_kinematics"
	# 	IPython.embed()
		# for line in f:
		# 	if i % sampling_rate == 0:
		# 		traj = np.array(line.split())
		# 		slave = traj[38:]
		# 		slave_left = traj[38:57]
		# 		slave_right = traj[57:]
		# 		if X is not None:
		# 			X = np.concatenate((X, slave.reshape(1, slave.shape[0])), axis = 0)
		# 		else:
		# 			X = slave.reshape(1, slave.shape[0])
		# 	i += 1
	return X.astype(np.float)


if __name__ == "__main__":
	forward_pass_entire_dataset()
