import pickle
import IPython
import argparse

# Author: Adithya Murali
# UC Berkeley, 2015

# Script to assist in manual segmentation of videos. Saves segment data, index and labels
# in Pickle file and txt document.

class Segmenter():
	def __init__(self, output_name, output_path):
		self.output_name = output_name
		self.map_index_data = {}
		self.map_index_labels = {}
		self.output_path = output_path
		pass
	def segment(self):
		# Specify segments and index
		exit = False
		i = 1
		while not exit:
			user_ret = raw_input("Enter new segment name: ")
			self.map_index_labels[i] = str(user_ret)
			self.map_index_data[i] = []
			i += 1
			user_ret = raw_input("Done with specifing segments?[y/n]")
			if user_ret == 'y':
				exit = True
		print "----------------Collecting Segments ----------------------"
		exit = False
		while not exit:
			self.print_all_index_labels()
			index = int(raw_input("Which index?"))
			start_frm = int(raw_input("Starting frame?"))
			end_frm = int(raw_input("End frame?"))
			new_segment = (start_frm, end_frm)
			segment_list = self.map_index_data[index]
			segment_list.append(new_segment)
			self.map_index_data[index] = segment_list
			user_ret = raw_input("Done with specifing segments?[y/n]")
			if user_ret == 'y':
				exit = True
		print self.map_index_data
		IPython.embed()

	def write_to_picke(self):
		pickle.dump(self.map_index_data, open(self.output_path + self.output_name + ".p", "wb"))

	def write_to_txt(self):
		f = open(self.output_path + self.output_name + ".txt", "wb")
		f.write("Index and Labels:\n")
		for index in self.map_index_labels:
			f.write(str(index) + " " + self.map_index_labels[index] + '\n')
		f.write("\n")
		f.write("Index and Data\n")
		for index in self.map_index_data:
			data = self.map_index_data[index]
			f.write(str(index) + " " + str(self.map_index_data[index]) + '\n')
		f.write("\n")
		f.write("~ Fin ~ ")
	def print_all_index_labels(self):
		for index in self.map_index_labels.keys():
			print str(index) + ": " + self.map_index_labels[index]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("output_name", help = "Please specify output name for pickle file and txt annotation")
	parser.add_argument("output_path", help = "Path to output files")
	args = parser.parse_args()
	seg = Segmenter(args.output_name, args.output_path)
	seg.segment()
	seg.write_to_txt()
	seg.write_to_picke()
