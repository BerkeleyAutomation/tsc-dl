import numpy as np
import cv2
import matplotlib.pyplot as plt
import IPython

def run_sift(cap):
	sift = cv2.SIFT()
	while(1):
		# img = cv2.imread('mum.JPG')
		# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# kp = sift.detect(gray,None)

		# IPython.embed()

		# img=cv2.drawKeypoints(gray,kp)


		ret, frame = cap.read()
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		kp = sift.detect(gray,None)
		img = cv2.drawKeypoints(gray,kp)
		cv2.imshow('frame',img)

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	cap = cv2.VideoCapture('/home/animesh/C3D/examples/c3d_feature_extraction/input/frm/pizza8/videos/cropped_scaled.avi')
	run_sift(cap)