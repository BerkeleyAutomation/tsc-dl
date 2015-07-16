import argparse
import IPython
import pickle

# Author: Adithya Murali
# UC Berkeley, 2015

# Script to autogenerate config files specifying input frames for C3D

def generate_files(args):

	input_list_path = "input/frm/"
	output_list_path = "output/c3d/"
	prototxt_path = "../prototxt/"

	input_list = open(prototxt_path + args.input_list_name, 'w')
	output_list = open(prototxt_path + args.output_list_name, 'w')
	fps = int(args.fps)

	if args.segments:
		all_segments = pickle.load(open(args.segments, "rb"))
		for index in all_segments:
			for elem in all_segments[index]:
				j = elem[0]
				while j <= (elem[1]):
					input_list.write(input_list_path + args.input_folder + "/ " + str(j) + " 0\n")
					output_list.write(output_list_path + args.output_folder + "/"+ str(j)+"\n")
					j += fps
	else:
		i = 1
		while i < (int(args.num_minutes) - 16):
			input_list.write(input_list_path + args.input_folder + "/ " + str(i) + " 0\n")
			output_list.write(output_list_path + args.output_folder + "/"+ str(i)+"\n")
			i += fps


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("input_list_name", help = "input list to be populated")
	parser.add_argument("input_folder", help = "Location of input images")
	parser.add_argument("output_list_name", help = "ouput list to be populated")
	parser.add_argument("output_folder", help = "Location of output features")
	parser.add_argument("num_minutes", help = "Number of 16-frame batches")
	parser.add_argument("fps", help = "Frame rate - Frames per second", default = 1)
	parser.add_argument("--segments", help = "Pickle file with segments")

	args = parser.parse_args()
	generate_files(args)