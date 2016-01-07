import os, sys
import IPython
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

import constants
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

def run_surf_frame(PATH_TO_IMAGE, n_features = 10):
	img = cv2.imread(PATH_TO_IMAGE)
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SURF(4000)
	# kp = sift.detect(gray, None)
	# img = cv2.drawKeypoints(gray, kp)
	kp = sift.detect(img, None)
	img2 = cv2.drawKeypoints(img, kp)
	cv2.imshow("frame", img2)
	return len(kp)

def generate_sift_features():
	list_of_demonstrations = ["plane_9",]
	for demonstration in list_of_demonstrations:
		print "SIFT for ", demonstration
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"

		X1 = None
		X2 = None
		n_features = 20
		sift = cv2.SIFT(nfeatures = n_features)

		start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)
		for frm in range(start, end + 1):
			# if ((frm % 3) == 0):
				PATH_TO_IMAGE = utils.get_full_image_path(constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + constants.CAMERA + "/", frm)

				print PATH_TO_IMAGE
				img = cv2.imread(PATH_TO_IMAGE)
				kp, des = sift.detectAndCompute(img, None)
				img = cv2.drawKeypoints(img, kp)
				cv2.imshow('sift',img)
				cv2.imwrite('../sift_images/' + demonstration + "/" + str(frm) +".jpg",img)

				vector1 = []
				vector2 = []
				kp.sort(key = lambda x: x.response, reverse = True)
				for kp_elem in kp:
					vector1 += [kp_elem.response, kp_elem.pt[0], kp_elem.pt[1], kp_elem.size, kp_elem.angle]
					vector2 += [kp_elem.pt[0], kp_elem.pt[1]]
				try:
					X1 = utils.safe_concatenate(X1, utils.reshape(np.array(vector1[:n_features * 5])))
					X2 = utils.safe_concatenate(X2, utils.reshape(np.array(vector2[:n_features * 2])))
				except ValueError as e:
					IPython.embed()

		pickle.dump(X1, open("sift_features/SIFT_" + demonstration + "_1.p", "wb"))
		pickle.dump(X2, open("sift_features/SIFT_" + demonstration + "_2.p", "wb"))

def run_sift_images():
	list_of_demonstrations = ["Suturing_E001",]
	for demonstration in list_of_demonstrations:
		print "SIFT for ", demonstration
		PATH_TO_ANNOTATION = constants.PATH_TO_DATA + constants.ANNOTATIONS_FOLDER + demonstration + "_" + str(constants.CAMERA) + ".p"

		start, end = utils.get_start_end_annotations(PATH_TO_ANNOTATION)
		for frm in range(start, end + 1):
			if ((frm % 3) == 0):
				PATH_TO_IMAGE = utils.get_full_image_path(constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER + demonstration + "_" + constants.CAMERA + "/", frm)

				print PATH_TO_IMAGE
				img = cv2.imread(PATH_TO_IMAGE)
				sift = cv2.SIFT(nfeatures = 50)
				kp, des = sift.detectAndCompute(img, None)
				img = cv2.drawKeypoints(img, kp)
				cv2.imshow('sift',img)
				cv2.imwrite('../sift_images/' + str(frm) +".jpg",img)


def run_sift(PATH_TO_DATA, count, n_features = 20):
	cap = cv2.VideoCapture(PATH_TO_DATA)
	sift = cv2.SIFT(nfeatures = n_features)
	i = 0
	X1 = None
	X2 = None
	IPython.embed()
	while(1):
		print str(count) + " "+ str(i)
		ret, frame = cap.read()
		if not ret:
			break;
		kp, des = sift.detectAndCompute(frame, None)

		img = cv2.drawKeypoints(frame, kp)

		cv2.imshow('sift',img)
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

	cap.release()
	cv2.destroyAllWindows()
	return X1, X2

if __name__ == "__main__":
	# list_of_demonstrations = ['Suturing_E001','Suturing_E002', 'Suturing_E003', 'Suturing_E004', 'Suturing_E005',
	# 'Suturing_D001','Suturing_D002', 'Suturing_D003', 'Suturing_D004', 'Suturing_D005',
	# 'Suturing_C001','Suturing_C002', 'Suturing_C003', 'Suturing_C004', 'Suturing_C005',
	# 'Suturing_F001','Suturing_F002', 'Suturing_F003', 'Suturing_F004', 'Suturing_F005']
	generate_sift_features()
	# j = 0
	# for demonstration in list_of_demonstrations:
	# 	X1, X2 = run_sift("/Users/adithyamurali/dev/DeepMilestones/sift_videos/"+ demonstration + ".mp4", j)
	# 	# pickle.dump(X1, open("SIFT_" + demonstration + "_1.p", "wb"))
	# 	# pickle.dump(X2, open("SIFT_" + demonstration + "_2.p", "wb"))
	# 	j += 1