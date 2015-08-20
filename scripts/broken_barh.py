"""
Make a "broken" horizontal bar plot, i.e., one with gaps
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import IPython
import pickle

import utils
import constants
import parser


# demonstration = "Needle_Passing_D001"

demonstration = "Suturing_E001"

PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + constants.CAMERA + ".p"

start, end = parser.get_start_end_annotations(constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER
	+ demonstration + "_" + constants.CAMERA + ".p")

def setup_manual_labels(segments):
	list_of_start_end = []
	list_of_colors = []

	for key in segments.keys():
		color = constants.color_map[key]

		for elem in segments[key]:
			list_of_start_end.append((elem[0], elem[1] - elem[0]))
			list_of_colors.append(color)

	return list_of_start_end, tuple(list_of_colors)

def setup_automatic_labels(list_of_frms, color):
	list_of_start_end = []
	list_of_colors = []
	for elem in list_of_frms:
		list_of_start_end.append((elem - 15, 30))
		list_of_colors.append(color)
	return list_of_start_end, tuple(list_of_colors)


#Needle passing examples
# list_of_frms_1 = [3679, 3772, 4465, 2170, 2215, 2233, 3154, 3169, 3889, 3904]

# list_of_frms_2 = [3679, 3772, 4465, 3154, 3163, 3889, 3908]


#Suturing Examples
list_of_frms_1 = [2596 , 2746, 2950, 1513, 1702, 1783, 1087, 1954, 205, 2227, 2386] #6282_4_PCA

list_of_frms_2 = [814, 1513, 205, 265, 289, 673, 778, 2386, 2596, 2746, 2950, 943, 1681, 1954] #5991_4_PCA

list_of_frms_3 = [1663, 2227, 3091, 3100, 3301, 1954, 1372, 1513, 673, 211, 265, 289, 784, 808] #4916_4_PCA  

list_of_frms_4 = [289, 265, 676, 1084, 1099, 1369, 1501, 1567] #6961_4_PCA

segments = pickle.load(open(PATH_TO_ANNOTATION, "rb"))
labels_manual, colors_manual = setup_manual_labels(segments)
labels_automatic_1, colors_automatic_1 = setup_automatic_labels(list_of_frms_1, "g")
labels_automatic_2, colors_automatic_2 = setup_automatic_labels(list_of_frms_2, "r")
labels_automatic_3, colors_automatic_3 = setup_automatic_labels(list_of_frms_3, "b")
labels_automatic_4, colors_automatic_4 = setup_automatic_labels(list_of_frms_4, "k")

fig, ax = plt.subplots()

ax.broken_barh(labels_manual, (50, 9), facecolors = colors_manual)
ax.broken_barh(labels_automatic_1, (40, 9), facecolors = colors_automatic_1)
ax.broken_barh(labels_automatic_2, (30, 9), facecolors = colors_automatic_2)
ax.broken_barh(labels_automatic_3, (20, 9), facecolors = colors_automatic_3)
ax.broken_barh(labels_automatic_4, (10, 9), facecolors = colors_automatic_4)

ax.set_ylim(5,65)
ax.set_xlim(0, end + 100) # Need to replace this with start and end frames
ax.set_xlabel('Frame number')
ax.set_yticks([15,25,35, 45])
ax.set_yticklabels(['Automatic4(E2)','Automatic3(E3)','Automatic2(E5)', 'Automatic1(E4)', 'Manual'])
ax.grid(True)

plt.show()