import numpy as np
import sys
import os
import IPython
import time
import random

sys.path.insert(0, "/home/animesh/DeepMilestones/scripts/")
import utils

PATH_TO_SUTURING_DATA = "/home/animesh/DeepMilestones/jigsaws/Suturing_video/"

FPS = 30

CONFIG_FILE = "meta_file_Suturing.txt"

VIDEO_FOLDER = "video/"

NEW_FRAMES_FOLDER = "frames/"

TRANSCRIPTIONS_FOLDER = "transcriptions/"

CROP_PARAMS = {"capture2": "\"crop=330:260:150:150\"", "capture1": "\"crop=330:260:200:170\""}

map_surgeme_label = {'G1': 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6, "G7": 7, "G8": 8, "G9": 9,
"G10": 10, "G12": 12, "G11": 11, "G13": 13, "G14": 14, "G15": 15, "G16": 16, "G17": 17}

train_val_split = 0.75

def generate_list_of_videos(config_file_name):
	list_of_videos = []
	with open(config_file_name, "rb") as f:
		for line in f:
			params = line.split()
			if len(params) != 0:
				list_of_videos.append(params[0])
	return list_of_videos

def convert_video_to_frames(list_of_videos):
	for video in list_of_videos:
		for camera in ["capture1", "capture2"]:
			os.chdir(PATH_TO_SUTURING_DATA)
			video_file_name = video + "_" + camera
			mkdir_path = NEW_FRAMES_FOLDER + video_file_name
			print "mkdir " + mkdir_path
			os.mkdir(mkdir_path)

			command = "cp " + VIDEO_FOLDER + video_file_name + ".avi " + mkdir_path
			print command
			os.system(command)

			command = PATH_TO_SUTURING_DATA + NEW_FRAMES_FOLDER + video_file_name
			print "cd " + command
			os.chdir(PATH_TO_SUTURING_DATA + NEW_FRAMES_FOLDER + video_file_name)

			command = "ffmpeg -i " + video_file_name + ".avi -filter:v " + CROP_PARAMS[camera] + " cropped.avi"
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

def generate_val_test_files(list_of_videos):
	list_of_data = []

	for video in list_of_videos:
		for camera in ["capture1", "capture2"]:
			os.chdir(PATH_TO_SUTURING_DATA)
			transcriptions_file = TRANSCRIPTIONS_FOLDER + video + ".txt"
			with open(transcriptions_file, "rb") as f:
				for line in f:
					line = line.split()
					start = int(line[0])
					end = int(line[1])
					surgeme = line[2]
					label = map_surgeme_label[surgeme]
					i = start
					while i <= end:
						data = PATH_TO_SUTURING_DATA + NEW_FRAMES_FOLDER + video + "_" + camera + "/" + utils.get_frame_fig_name(i) + " "+ str(label) + " \n"
						list_of_data.append(data)
						i += 1
	random.shuffle(list_of_data)

	N = len(list_of_data)

	train_file_name = PATH_TO_SUTURING_DATA + "train.txt"
	val_file_name = PATH_TO_SUTURING_DATA + "val.txt"

	if not os.path.exists(train_file_name):
		train = open(train_file_name, 'w')
		for i in range(int(N * train_val_split)):
			train.write(list_of_data[i])
		train.close()

	if not os.path.exists(val_file_name):
		val = open(val_file_name, 'w')
		for i in range(int(N * train_val_split), N):
			val.write(list_of_data[i])
		val.close()

	pass

if __name__ == "__main__":
	list_of_videos = generate_list_of_videos(PATH_TO_SUTURING_DATA + CONFIG_FILE)
	# convert_video_to_frames(list_of_videos)
	generate_val_test_files(list_of_videos)
