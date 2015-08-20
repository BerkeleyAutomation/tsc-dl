import numpy as np
import cv2
import matplotlib.pyplot as plt
import IPython
import pickle

import utils

def min_kp_SIFT(PATH_TO_DATA):
	print "Calculating #of features for video: ", PATH_TO_DATA
	cap = cv2.VideoCapture(PATH_TO_DATA)
	sift = cv2.SIFT(nfeatures = 10)
	result = []
	i = 0
	while(1):
		ret, frame = cap.read()
		if not ret:
			break;
		kp, des = sift.detectAndCompute(frame, None)
		print i
		result.append(len(kp))
		i += 1

	cap.release()
	return min(result)

def run_sift(PATH_TO_DATA, count, n_features = 20):

	cap = cv2.VideoCapture(PATH_TO_DATA)
	sift = cv2.SIFT(nfeatures = n_features)
	i = 0
	X1 = None
	X2 = None
	while(1):
		print str(count) + " "+ str(i)
		ret, frame = cap.read()
		if not ret:
			break;
		kp, des = sift.detectAndCompute(frame, None)

		vector1 = []
		vector2 = []
		kp.sort(key = lambda x: x.response, reverse = True)
		for kp_elem in kp:
			vector1 += [kp_elem.response, kp_elem.pt[0], kp_elem.pt[1], kp_elem.size, kp_elem.angle]
			vector2 += [kp_elem.pt[0], kp_elem.pt[1]]
		# vector2 = utils.reshape(des.flatten())

		try:
			X1 = utils.safe_concatenate(X1, utils.reshape(np.array(vector1[:n_features * 5])))
			X2 = utils.safe_concatenate(X2, utils.reshape(np.array(vector2[:n_features * 2])))
		except ValueError as e:
			IPython.embed()
		i += 1

	cap.release()
	cv2.destroyAllWindows()
	return X1, X2

if __name__ == "__main__":
	list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
	'Suturing_D001','Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
	'Suturing_C001','Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
	'Suturing_F001','Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']

	# list_of_demonstrations = ["Suturing_E001"]

	j = 0
	for demonstration in list_of_demonstrations:
		X1, X2 = run_sift("/Users/adithyamurali/dev/DeepMilestones/sift_videos/"+ demonstration + ".mp4", j)
		pickle.dump(X1, open("SIFT_" + demonstration + "_1.p", "wb"))
		pickle.dump(X2, open("SIFT_" + demonstration + "_2.p", "wb"))
		j += 1
