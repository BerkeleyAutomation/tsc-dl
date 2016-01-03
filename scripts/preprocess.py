#!/usr/bin/env python

import numpy as np
import sys
import os
import IPython

import utils
import constants
import parser
import time

def convert_video_to_frames(list_of_videos, cropping_params):

	for video_file_name in list_of_videos:

			command = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + video_file_name
			print "cd " + command
			os.chdir(constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + video_file_name)

			command = "ffmpeg -i " + video_file_name + ".avi -filter:v " + cropping_params[video_file_name] + " cropped.avi"
			print command
			os.system(command)

			command = "ffmpeg -i cropped.avi -vf scale=640:480 cropped_scaled.avi"
			print command
			os.system(command)

			command = "ffmpeg -i cropped_scaled.avi -r " + 	str(30) + " ./%6d.jpg"
			print command
			os.system(command)

			time.sleep(1)
	pass

def preprocess(list_of_demonstrations):
	camera = constants.CAMERA

	for demonstration in list_of_demonstrations:
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + camera + ".p"
		start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)

		OLD_FRM_PATH = constants.PATH_TO_DATA + "frames_unprocessed/" + demonstration + "_" + camera + "/"
		NEW_FRM_PATH = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + camera + "/"

		command = "mkdir " + NEW_FRM_PATH
		print command
		os.mkdir(NEW_FRM_PATH)

		for frm in range(start, end + 1):
			OLD_FRM_NAME = utils.get_full_image_path(OLD_FRM_PATH, frm)

			NEW_FRM_NAME = utils.get_full_image_path(NEW_FRM_PATH, frm)
			NEW_FRM_NAME_UNSCALED = utils.get_full_image_path(NEW_FRM_PATH + "unscaled_", frm)

			command = "ffmpeg -i " + OLD_FRM_NAME + " -filter:v " + constants.CROP_PARAMS[camera] + " " + NEW_FRM_NAME_UNSCALED
			print command
			os.system(command)

			command = "ffmpeg -i " + NEW_FRM_NAME_UNSCALED + " -vf scale=640:480 " + NEW_FRM_NAME
			print command
			os.system(command)

			command = "rm " + NEW_FRM_NAME_UNSCALED
			print command
			os.system(command)	

if __name__ == "__main__":

	# list_of_demonstrations = ["plane_5",]
	# list_of_demonstrations = ["plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]

	# list_of_demonstrations = ["people_0", "people_1", "people_2", "people_3", "people_4", "people_5", "people_6"]

	# list_of_demonstrations = ["lego_2", "lego_3", "lego_4", "lego_5", "lego_6", "lego_7"]

	# list_of_demonstrations = ["people2_2", "people2_3", "people2_4", "people2_5", "people2_6", "people2_7"]

	list_of_demonstrations = ["Needle_Passing_B001_capture1", "Needle_Passing_B002_capture1", "Needle_Passing_B003_capture1",
	"Needle_Passing_B004_capture1", "Needle_Passing_C001_capture1", "Needle_Passing_C002_capture1", "Needle_Passing_C003_capture1",
	"Needle_Passing_C004_capture1","Needle_Passing_C005_capture1", "Needle_Passing_F001_capture1", "Needle_Passing_F003_capture1",
	"Needle_Passing_F004_capture1", "Needle_Passing_H002_capture1", "Needle_Passing_H004_capture1","Needle_Passing_H005_capture1",
	"Needle_Passing_I002_capture1", "Needle_Passing_I003_capture1", "Needle_Passing_I004_capture1", "Needle_Passing_I005_capture1"]

	cropping_params = {"Needle_Passing_B001_capture1": "crop=434:366:125:107", "Needle_Passing_B002_capture1": "crop=437:368:109:98",
	"Needle_Passing_B003_capture1": "crop=439:367:122:104", "Needle_Passing_B004_capture1": "crop=441:367:118:97",
	"Needle_Passing_C001_capture1": "crop=371:392:152:82", "Needle_Passing_C002_capture1": "crop=364:392:156:82",
	"Needle_Passing_C003_capture1": "crop=362:393:156:82", "Needle_Passing_C004_capture1": "crop=364:390:152:82", 
	"Needle_Passing_C005_capture1": "crop=363:392:156:82", "Needle_Passing_D001_capture1": "crop=415:398:96:83",
	"Needle_Passing_D002_capture1": "crop=414:398:96:83", "Needle_Passing_D003_capture1": "crop=413:398:96:83",
	"Needle_Passing_D004_capture1": "crop=414:398:96:83", "Needle_Passing_D005_capture1": "crop=389:398:105:83", 
	"Needle_Passing_E001_capture1": "crop=433:396:142:1", "Needle_Passing_E003_capture1": "crop=422:361:112:84", 
	"Needle_Passing_E004_capture1": "crop=422:361:112:84", "Needle_Passing_E005_capture1": "crop=422:361:112:84", 
	"Needle_Passing_F001_capture1": "crop=202:188:58:40", "Needle_Passing_F003_capture1": "crop=202:189:58:40", 
	"Needle_Passing_F004_capture1": "crop=202:188:58:40", "Needle_Passing_H002_capture1": "crop=193:188:82:41",
	"Needle_Passing_H004_capture1": "crop=191:188:84:41", "Needle_Passing_H005_capture1": "crop=191:188:84:41", 
	"Needle_Passing_I002_capture1": "crop=352:364:108:68", "Needle_Passing_I003_capture1": "crop=178:180:68:42",
	"Needle_Passing_I004_capture1": "crop=351:363:153:83", "Needle_Passing_I005_capture1": "crop=347:363:153:83"} 



	convert_video_to_frames(list_of_demonstrations, cropping_params)
	# preprocess(list_of_demonstrations)
	pass