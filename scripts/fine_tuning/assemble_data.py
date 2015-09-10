#!/usr/bin/env python

import numpy as np
import sys
import os
import IPython
import time
import random
from sklearn.cross_validation import train_test_split

sys.path.insert(0, "/home/animesh/DeepMilestones/scripts/")
import utils
import constants
import parser

FPS = 30

def convert_video_to_frames(list_of_videos):

	for video in list_of_videos:
		# for camera in ["capture1", "capture2"]:
		for camera in ["capture1",]:
			# os.chdir(constants.PATH_TO_DATA)
			video_file_name = video + "_" + camera
			# mkdir_path = constants.NEW_FRAMES_FOLDER + video_file_name
			# print "mkdir " + mkdir_path
			# os.mkdir(mkdir_path)

			# command = "cp " + constants.VIDEO_FOLDER + video_file_name + ".avi " + mkdir_path
			# print command
			# os.system(command)

			command = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + video_file_name
			print "cd " + command
			os.chdir(constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + video_file_name)

			command = "ffmpeg -i " + video_file_name + ".avi -filter:v " + constant.CROP_PARAMS[camera] + " cropped.avi"
			print command
			os.system(command)

			command = "ffmpeg -i cropped.avi -vf scale=640:480 cropped_scaled.avi"
			print command
			os.system(command)

			command = "ffmpeg -i cropped_scaled.avi -r " + 	str(FPS) + " ./%6d.jpg"
			print command
			os.system(command)

			time.sleep(1)
	pass


def write_to_file(file_name, data):
	if not os.path.exists(file_name):
		file = open(file_name, 'w')
		for elem in data:
			file.write(elem)
		file.close()

def generate_train_val_test_files(list_of_videos):
	list_of_data = []

	for video in list_of_videos:
		for camera in ["capture1", "capture2"]:
			os.chdir(constants.PATH_TO_DATA)
			transcriptions_file = TRANSCRIPTIONS_FOLDER + video + ".txt"
			with open(transcriptions_file, "rb") as f:
				for line in f:
					line = line.split()
					start = int(line[0])
					end = int(line[1])
					surgeme = line[2]
					label = constants.map_surgeme_label[surgeme]
					i = start
					while i <= end:
						data = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + video + "_" + camera + "/" + utils.get_frame_fig_name(i) + " "+ str(label) + " \n"
						list_of_data.append(data)
						i += 1
	random.shuffle(list_of_data)

	N = len(list_of_data)

	train, test = train_test_split(list_of_data, test_size = constants.train_test_ratio)

	val = test[:len(test)/2]
	test = test[len(test)/2:]

	train_file_name = constants.PATH_TO_DATA + "train.txt"
	val_file_name = constants.PATH_TO_DATA + "val.txt"
	test_file_name = constants.PATH_TO_DATA + "test.txt"

	IPython.embed()

	write_to_file(train_file_name, train)
	write_to_file(val_file_name, val)
	write_to_file(test_file_name, test)

if __name__ == "__main__":
	# list_of_videos = parser.generate_list_of_videos(constants.PATH_TO_DATA + constants.CONFIG_FILE)
	list_of_videos = ["Needle_Passing_E001", "Needle_Passing_E003", "Needle_Passing_E004", "Needle_Passing_E005",
	"Needle_Passing_D001", "Needle_Passing_D002","Needle_Passing_D003", "Needle_Passing_D004", "Needle_Passing_D005"]
	convert_video_to_frames(list_of_videos)
	# generate_train_val_test_files(list_of_videos)
