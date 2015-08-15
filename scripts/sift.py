import os, sys

# SIFT and SURF are only available for OpenCV 2.4.9
os.chdir(os.path.expanduser('~/opencv_2.4.9/opencv-2.4.9/lib/'))
sys.path.append(os.path.expanduser('~/opencv_2.4.9/opencv-2.4.9/lib/python2.7/dist-packages'))

import cv2
import numpy as np
import utils

def min_kp_SIFT(PATH_TO_DATA):
	print("Calculating #of features for video: " + PATH_TO_DATA)
	cap = cv2.VideoCapture(PATH_TO_DATA)
	sift = cv2.SIFT(nfeatures = 10)
	result = []
	i = 0
	while(1):
		ret, frame = cap.read()
		if not ret:
			break;
		kp, des = sift.detectAndCompute(frame, None)
		print(i)
		result.append(len(kp))
		i += 1

	cap.release()
	return min(result)

def run_sift(PATH_TO_DATA):
	print("SIFT for video: " + PATH_TO_DATA)
	# n_features = min_kp_SIFT(PATH_TO_DATA)
	n_features = 10
	cap = cv2.VideoCapture(PATH_TO_DATA)
	sift = cv2.SIFT(nfeatures = n_features)
	i = 0
	X1 = None
	X2 = None
	while(1):
		print(i)
		ret, frame = cap.read()
		if not ret:
			break;
		kp, des = sift.detectAndCompute(frame, None)

		img = cv2.drawKeypoints(frame,kp)

		cv2.imshow('sift',img)
		vector1 = []
		kp.sort(key = lambda x: x.response, reverse = True)
		for kp_elem in kp:
			vector1 += [kp_elem.response, kp_elem.pt[0], kp_elem.pt[1], kp_elem.size, kp_elem.angle]

		vector2 = utils.reshape(des.flatten())

		X1 = utils.safe_concatenate(X1, utils.reshape(np.array(vector1)))
		X2 = utils.safe_concatenate(X2, vector2)
		i += 1

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	run_sift('/home/animesh/DeepMilestones/jigsaws/Suturing_video/frames/Suturing_E005_capture2/cropped_scaled.avi')