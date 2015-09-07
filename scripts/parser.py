#!/usr/bin/env python
import IPython
import pickle
import numpy as np
import scipy.io

import constants
import utils

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

def parse_annotations(list_of_demonstrations = None):
	"""
	Note that left and right cameres have same transcriptions/annotations
	"""
	if not list_of_demonstrations:
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

def parse_kinematics(PATH_TO_KINEMATICS_DATA, PATH_TO_ANNOTATION, fname):
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
	if constants.SIMULATION:
		mat = scipy.io.loadmat(PATH_TO_KINEMATICS_DATA + fname)
		X = mat['x_traj']
		X = X.T
		# X = pickle.load(open(PATH_TO_KINEMATICS_DATA + fname + ".p", "rb"))
	elif constants.TASK_NAME == "plane" or constants.TASK_NAME == "lego":
		print "-- Parsing Kinematics for ", fname
		trajectory = pickle.load(open(PATH_TO_KINEMATICS_DATA + fname + ".p", "rb"))
		for frm in range(start, end + 1):
			try:
				traj_point = trajectory[frm - start]
			except IndexError as e:
				print e
				IPython.embed()
			# vector = list(traj_point.position[16:-12]) + list(traj_point.velocity[16:-12])
			X = utils.safe_concatenate(X, utils.reshape(traj_point))

	else:
		X = None
		all_lines = open(PATH_TO_KINEMATICS_DATA + fname + ".txt", "rb").readlines()
		i = start - 1
		if i < 0:
			i = 0 
		while i < end:
			traj = np.array(all_lines[i].split())
			slave = traj[constants.KINEMATICS_DIM:]
			X = utils.safe_concatenate(X, utils.reshape(slave))
			i += 1
	return X.astype(np.float)

def get_kinematic_features(demonstration):
	"""
	Marshalls request to format needed for parse_kinematics.
	"""
	return parse_kinematics(constants.PATH_TO_KINEMATICS, constants.PATH_TO_DATA
		+ constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA +".p", demonstration)

if __name__ == "__main__":

	# list_of_demonstrations = ["0001_01", "0001_02", "0001_03", "0001_04", "0001_05"]

	# list_of_demonstrations = ["baseline2_000_01", "baseline2_000_02", "baseline2_000_03", "baseline2_000_04", "baseline2_000_05"]

	# list_of_demonstrations = ["baseline2_010_01", "baseline2_010_02", "baseline2_010_03", "baseline2_010_04", "baseline2_010_05"]

	# list_of_demonstrations = ["100_01", "100_02", "100_03", "100_04", "100_05"]

	list_of_demonstrations = ["plane_5", "plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

	parse_annotations(list_of_demonstrations)

	# X = parse_kinematics(constants.PATH_TO_KINEMATICS, constants.PATH_TO_DATA + "annotations/0001_02_capture1.p", "0001_02.mat")
	# IPython.embed()
	pass