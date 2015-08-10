import numpy as np
import cv2
import matplotlib.pyplot as plt
import IPython

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

def run_sift(PATH_TO_DATA):
	print "SIFT for video: ", PATH_TO_DATA
	# n_features = min_kp_SIFT(PATH_TO_DATA)
	n_features = 10
	cap = cv2.VideoCapture(PATH_TO_DATA)
	sift = cv2.SIFT(nfeatures = n_features)
	i = 0
	X1 = None
	X2 = None
	while(1):
		print i
		ret, frame = cap.read()
		if not ret:
			break;
		kp, des = sift.detectAndCompute(frame, None)

		vector1 = []
		kp.sort(key = lambda x: x.response, reverse = True)
		for kp_elem in kp:
			vector1 += [kp_elem.response, kp_elem.pt[0], kp_elem.pt[1], kp_elem.size, kp_elem.angle]

		vector2 = utils.reshape(des.flatten())

		print len(vector1)
		print vector2.shape
		IPython.embed()
		X1 = utils.safe_concatenate(X1, utils.reshape(np.array(vector1)))
		X2 = utils.safe_concatenate(X2, vector2)
		i += 1

	IPython.embed()
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	run_sift('../images/cropped_scaled.mp4')