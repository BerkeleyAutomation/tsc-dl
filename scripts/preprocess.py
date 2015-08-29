#!/usr/bin/env python

import numpy as np
import sys
import os
import IPython

import utils
import constants
import parser

def preprocess(list_of_demonstrations):
	camera = constants.CAMERA

	for demonstration in list_of_demonstrations:
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + camera + ".p"
		start, end = parser.get_start_end_annotations(PATH_TO_ANNOTATION)

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

	# list_of_demonstrations = ["plane_0",]
	list_of_demonstrations = ["plane_1", "plane_2", "plane_3", "plane_4", "plane_5",
		"plane_6", "plane_7", "plane_8", "plane_9", "plane_10"]
	preprocess(list_of_demonstrations)
	pass