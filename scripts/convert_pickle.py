import numpy as np
import pickle

import constants
import utils
import parser

list_of_joint_states = ["plane_3_js.p", "plane_4_js.p", "plane_5_js.p",
		"plane_6_js.p", "plane_7_js.p", "plane_8_js.p", "plane_9_js.p", "plane_10_js.p"]

list_of_trajectories = ["plane_3.p", "plane_4.p", "plane_5.p",
		"plane_6.p", "plane_7.p", "plane_8.p", "plane_9.p", "plane_10.p"]

list_of_annotations = ["plane_3_capture2.p", "plane_4_capture2.p", "plane_5_capture2.p",
		"plane_6_capture2.p", "plane_7_capture2.p", "plane_8_capture2.p", "plane_9_capture2.p", "plane_10_capture2.p"]

for i in range(len(list_of_annotations)):
	print list_of_annotations[i], list_of_joint_states[i], list_of_trajectories[i]
	start, end = utils.get_start_end_annotations(constants.PATH_TO_DATA + "annotations/" + list_of_annotations[i])
	X = None
	trajectory = pickle.load(open(constants.PATH_TO_KINEMATICS + list_of_joint_states[i], "rb"))
	for frm in range(start, end + 1):
		traj_point = trajectory[frm]
		print traj_point.velocity[16:-12]
		vector = list(traj_point.position[16:-12]) + list(traj_point.velocity[16:-12])
		X = utils.safe_concatenate(X, utils.reshape(np.array(vector)))
	# pickle.dump(X, open(constants.PATH_TO_KINEMATICS + list_of_trajectories[i],"wb"))