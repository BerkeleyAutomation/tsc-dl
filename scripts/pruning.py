#!/usr/bin/env python

import IPython
import pickle
import numpy as np

import constants
import utils

experts = ["E001", "E002", "E003", "E004", "E005", "D001", "D002", "D003", "D004", "D005"]

intermediates = ["F001", "F002", "F003", "F004", "F005", "C001", "C002", "C003", "C004", "C005"]

novices = ["G001", "G002", "G003", "G004", "G005", "B001", "B002", "B003", "B004", "B005",
"I001", "I002", "I003", "I004", "I005", "H001", "H002", "H003", "H004", "H005"]

def weighted_score(list_of_demonstrations, list_of_frm_demonstrations):
	"""
	Implements weighted pruning for given demonstrations represented in list_of_frm_demonstrations.
	Returns weighted score.
	"""

	if constants.TASK_NAME not in ["suturing", "needle_passing"]:
		return None

	if not constants.WEIGHTED_PRUNING_MODE:
		return None

	N = float(len(list_of_demonstrations))
	uniform_weight = 1/N
	map_demonstration2weight = {}

	for demonstration in list_of_demonstrations:
		# Base weight
		weight = uniform_weight

		# Stripped demonstration task (Suturing_, Needle_passing_, etc.) from demonstration name
		demonstration_name = demonstration.split("_")[-1]

		if demonstration_name in experts:
			weight *= constants.WEIGHT_EXPERT
		elif demonstration_name in intermediates:
			weight *= constants.WEIGHT_INTERMEDIATE
		else:
			if demonstration_name not in novices:
				print "ERROR: Unidentified Demonstration"
				IPython.embed()
		map_demonstration2weight[demonstration] = weight

	normalization_factor = sum(map_demonstration2weight.values())

	#Weight normalization
	for demonstration in list_of_demonstrations:
		weight = map_demonstration2weight[demonstration]
		map_demonstration2weight[demonstration] = weight/float(normalization_factor)

	score = 0.0
	for demonstration in set(list_of_frm_demonstrations):
		score += map_demonstration2weight[demonstration]

	return score