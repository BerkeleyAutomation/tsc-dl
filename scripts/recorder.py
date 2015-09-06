import sys
# sys.path.append()

import roslib
import rospy
import cPickle as pickle
import numpy as np
import os
import IPython
import time
import tf

from geometry_msgs.msg import PoseStamped, PoseArray
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Float32

import cv
import cv2
import cv_bridge
import numpy as np
import signal

import constants

def get_frame_fig_name(frm_num):
    """
    Useful for parsing frames and loading into memory.
    """
    if len(str(frm_num)) == 1:
        return "00000" + str(frm_num) + ".jpg"
    elif len(str(frm_num)) == 2:
        return "0000" + str(frm_num) + ".jpg"
    elif len(str(frm_num)) == 3:
        return "000" + str(frm_num) + ".jpg"
    elif len(str(frm_num)) == 4:
        return "00" + str(frm_num) + ".jpg"
    else:
        pass

class Recording(object):

    def __init__(self, trial_name):

        # Preparing folders
        self.task_name = constants.TASK_NAME
        self.listener = tf.TransformListener()

        self.trial_name = trial_name
        self.kinematics_folder = constants.PATH_TO_KINEMATICS
        self.video_folder = constants.PATH_TO_DATA + constants.NEW_FRAMES_FOLDER

        # Make folders for frames
        command = self.video_folder + self.task_name + "_" + self.trial_name + "_capture1/"
        print "mkdir " + command
        os.makedirs(command)
        command = self.video_folder + self.task_name + "_" + self.trial_name + "_capture2/"
        print "mkdir " + command
        os.makedirs(command)

        self.data = []
        self.images = []
        self.frequency = 10 # max save framerate is 10

        # Data to record
        self.left_image = None
        self.right_image = None
        self.psm1_gripper = None
        self.psm2_gripper = None
        self.psm1_pose = None
        self.psm2_pose = None
        self.joint_state = None

        # Subscribers for images
        rospy.Subscriber("/wide_stereo/left/image_rect_color", Image, self.left_image_callback, queue_size=1)
        rospy.Subscriber("/wide_stereo/right/image_rect_color", Image, self.right_image_callback, queue_size=1)

        # Subscribers for kinematics
        rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        self.bridge = cv_bridge.CvBridge()
        self.isRecording = True

        signal.signal(signal.SIGINT, self.signal_handler)


    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def joint_state_callback(self, msg):
        self.joint_state = msg

    def signal_handler(self, signum, something):
        rospy.loginfo("Saving data to pickle file.")
        try:
            print "Saving pickle file to ", self.kinematics_folder + self.task_name +"_" + self.trial_name + ".p" 
            pickle.dump(self.data, open(self.kinematics_folder + self.task_name +"_" + self.trial_name + ".p", "wb"))

        except Exception as e:
            print "Exception: ", e
            rospy.logwarn('Failed to save registration')
            IPython.embed()
        sys.exit()

    def start_recording(self):

        print "Recorder Loop"

        while self.left_image is None or self.right_image is None:
            pass

        start = time.clock()

        frm = 0
        for i in range(9999999):
            print frm
            rospy.sleep(1.0/self.frequency)

            cv2.imwrite(self.video_folder + self.task_name + "_" + self.trial_name + "_capture1/" + str(get_frame_fig_name(frm)), self.left_image)
            cv2.imwrite(self.video_folder + self.task_name + "_" + self.trial_name + "_capture2/" + str(get_frame_fig_name(frm)), self.right_image)

            (trans,rot) = listener.lookupTransform('/turtle2', '/turtle1', rospy.Time(0))
            # Gripper angle is r_gripper_joint
            self.data.append(self.joint_state)
            frm += 1

if __name__ == "__main__":
    rospy.init_node("recorder_node")
    if len(sys.argv) < 2:
        print "ERROR: Please provide correct arguments"
        sys.exit()
    recording = Recording(sys.argv[1])
    recording.start_recording()