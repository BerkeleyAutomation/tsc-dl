import numpy as np
import cv2
import matplotlib.pyplot as plt
import IPython

def run_sift(cap):
	sift = cv2.SURF()
	i = 0
	while(1):
		# img = cv2.imread('mum.JPG')
		# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# kp = sift.detect(gray,None)

		# IPython.embed()

		# img=cv2.drawKeypoints(gray,kp)


		ret, frame = cap.read()
		# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		kp, des = sift.detectAndCompute(frame,None)
		print i, len(kp), len(des)
		img = cv2.drawKeypoints(frame,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite("../images/savefig/"+ str(i)+".jpg", img)
		i += 1

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	cap = cv2.VideoCapture('../images/cropped_scaled.mp4')
	run_sift(cap)