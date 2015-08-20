import numpy as np
import cv2
import argparse
import sys
import matplotlib.pyplot as plt
import IPython

import parser
import utils
import constants

def run_video_with_bsub(cap, func, kernel = None, params = None):
	if params is not None:
		fgbg = func(params[0], params[1], params[2], params[3])
	else:
		fgbg = func()
	SAVE_PATH = constants.PATH_TO_DATA + constants.NEW_BGSUB_FOLDER + "Suturing_E003_capture2/"
	i = 1
	while(1):
		ret, frame = cap.read()
		print i
		fgmask = fgbg.apply(frame)
		mask_rbg = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
		draw = frame & mask_rbg
		if frame == None and mask_rbg == None:
			sys.exit()
		if kernel is not None:
			fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

		# cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
		# cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
		cv2.imshow('frame', draw)
		cv2.imwrite(SAVE_PATH + utils.get_frame_fig_name(i), draw)
		k = cv2.waitKey(30) & 0xff
		i += 1
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("video", help = "A: Suturing: B: Pizza making")
	parser.add_argument("type", help = "1: MOG 2: MOG2 3: Ken's algorithm")
	args = parser.parse_args()
	cap = None
	if args.video == 'A':
		cap = cv2.VideoCapture('/home/animesh/DeepMilestones/jigsaws/Suturing_video/frames/Suturing_E003_capture2/cropped_scaled.avi')
	elif args.video == 'B':
		cap = cv2.VideoCapture('/home/animesh/C3D/examples/c3d_feature_extraction/input/frm/pizza8/videos/cropped_scaled.avi')
	else:
		print "Invalid video type"
		sys.exit()

	if (int(args.type) == 1):
		params = (500, 10, 0.9, 1)
		run_video_with_bsub(cap, cv2.BackgroundSubtractorMOG, params = None)
	elif (int(args.type) == 2):
		run_video_with_bsub(cap, cv2.BackgroundSubtractorMOG2)
	elif (int(args.type) == 3):
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		run_video_with_bsub(cap, cv2.createBackgroundSubtractorGMG, kernel = kernel)
	else:
		print "Error Type"
		sys.exit()