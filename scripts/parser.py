import IPython
import pickle
import argparse
import constants
import utils

def generate_list_of_videos(config_file_name):
	list_of_videos = []
	with open(config_file_name, "rb") as f:
		for line in f:
			params = line.split()
			if len(params) != 0:
				list_of_videos.append(params[0])
	return list_of_videos

def convert_trabscription_to_annotation(PATH_TO_TRANSCRIPTION, PATH_TO_ANNOTATION, base_fname):
	segments = {}
	with open(PATH_TO_TRANSCRIPTION + base_fname + ".txt", "rb") as f:
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

	pickle.dump(segments, open(PATH_TO_ANNOTATION + base_fname + "_capture1.p", "wb"))
	pickle.dump(segments, open(PATH_TO_ANNOTATION + base_fname + "_capture2.p", "wb"))

def parse_annotations():
	list_of_videos = generate_list_of_videos(constants.PATH_TO_SUTURING_DATA + constants.CONFIG_FILE)
	# Note that left and right cameres have same transcriptions/annotations
	for video in list_of_videos:
		convert_trabscription_to_annotation(constants.PATH_TO_SUTURING_DATA + constants.TRANSCRIPTIONS_FOLDER,
			constants.PATH_TO_SUTURING_DATA + constants.ANNOTATIONS_FOLDER, video)

if __name__ == "__main__":
	parse_annotations()