import pickle
import numpy as np
import utils
import constants
import IPython

# list_of_demonstrations = ["plane_5", "plane_6","plane_7","plane_8","plane_9", "plane_10"]
list_of_demonstrations = ["lego_2", "lego_3", "lego_4", "lego_5", "lego_6", "lego_7"]

POSE_DIM = 19
JOINTS_DIM = 24

for demonstration in list_of_demonstrations:
	X = pickle.load(open(constants.PATH_TO_KINEMATICS + demonstration + ".p", "rb"))
	assert X.shape[1] == (POSE_DIM + JOINTS_DIM)

	X_transpose = X.T

	X_POSE = X_transpose[:POSE_DIM].T
	X_JOINTS = X_transpose[:JOINTS_DIM].T

	assert X_POSE.shape[1] == POSE_DIM
	assert X_JOINTS.shape[1] == JOINTS_DIM

	assert X_POSE.shape[0] == X.shape[0]
	assert X_JOINTS.shape[0] == X.shape[0]

	pickle.dump(X_POSE, open(constants.PATH_TO_KINEMATICS + demonstration + "_POSE.p", "wb"))
	pickle.dump(X_JOINTS, open(constants.PATH_TO_KINEMATICS + demonstration + "_JOINTS.p", "wb"))