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
import utils
import time

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Float32
from tf import transformations

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

        self.record_kinematics = False
        if self.task_name in ["plane", "lego"]:
            self.record_kinematics = True

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

        self.data = None
        self.frequency = 10 # max save framerate is 10
        self.period = 1.0/self.frequency

        # Data to record
        self.left_image = None
        self.right_image = None
        self.psm1_gripper = None
        self.psm2_gripper = None
        self.psm1_pose = None
        self.psm2_pose = None
        self.joint_state = None
        self.listener = tf.TransformListener()

        # Subscribers for images
        rospy.Subscriber("/wide_stereo/left/image_rect_color", Image, self.left_image_callback, queue_size=1)
        rospy.Subscriber("/wide_stereo/right/image_rect_color", Image, self.right_image_callback, queue_size=1)

        if self.record_kinematics:
            # Subscribers for kinematics
            rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        self.bridge = cv_bridge.CvBridge()
        self.r_l = 0
        self.r_r = 0

        signal.signal(signal.SIGINT, self.signal_handler)

    def left_image_callback(self, msg):
        self.r_l += 1
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def right_image_callback(self, msg):
        self.r_r += 1
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def joint_state_callback(self, msg):
        self.joint_state = msg

    def save_and_quit(self):
        if self.record_kinematics:
            rospy.loginfo("Saving data to pickle file.")
            try:
                print "Saving pickle file to ", self.kinematics_folder + self.task_name +"_" + self.trial_name + ".p" 
                pickle.dump(self.data, open(self.kinematics_folder + self.task_name +"_" + self.trial_name + ".p", "wb"))
            except Exception as e:
                print "Exception: ", e
                rospy.logwarn('Failed to save registration')
                IPython.embed()
        print "Fin"
        sys.exit()

    def signal_handler(self, signum, something):
        self.save_and_quit()

    def start_recording(self):

        print "Recorder Loop"
        while self.left_image is None or self.right_image is None:
            pass

        if self.record_kinematics:
            while (1):
                try:
                    (trans,rot) = self.listener.lookupTransform('/r_gripper_tool_frame', '/base_link', rospy.Time(0))
                    break
                except (tf.ExtrapolationException):
                    print "ExtrapolationException"
                    rospy.sleep(0.1)
                    continue

        frm = 0
        wait_thresh = 0
        prev_r_l = self.r_l
        prev_r_r = self.r_r

        trans_vel = np.array([0.0, 0.0, 0.0])
        rot_vel = np.array([0.0, 0.0, 0.0])

        prev_trans = None
        prev_rot = None

        for i in range(9999999):
            print frm
            rospy.sleep(self.period)

            start = time.time()

            cv2.imwrite(self.video_folder + self.task_name + "_" + self.trial_name + "_capture1/" + str(get_frame_fig_name(frm)), self.left_image)
            cv2.imwrite(self.video_folder + self.task_name + "_" + self.trial_name + "_capture2/" + str(get_frame_fig_name(frm)), self.right_image)

            if self.record_kinematics:

                (trans, quaternion) = self.listener.lookupTransform('/r_gripper_tool_frame', '/base_link', rospy.Time(0))
                r_matrix = utils.quaternion2rotation(quaternion)
                rot = transformations.euler_from_matrix(r_matrix)
                r_gripper_angle = self.joint_state.position[-17]

                if frm != 0:
                    trans_vel = (trans - prev_trans) / self.period
                    rot_vel = (rot - prev_rot) / self.period

                prev_trans = np.array(trans)
                prev_rot = np.array(rot)

                js_pos = self.joint_state.position[16:-12]
                js_vel = self.joint_state.velocity[16:-12]

                W = list(trans) + list(r_matrix.flatten()) + list(trans_vel) + list(rot_vel)

                # Gripper angle is r_gripper_joint
                W.append(r_gripper_angle)

                W = W + list(js_pos) + list(js_vel)

                self.data = utils.safe_concatenate(self.data, utils.reshape(np.array(W)))

            frm += 1

            if ((self.r_l == prev_r_l) and (self.r_r == prev_r_r)):
                print "Not recording anymore?"
                wait_thresh += 1
                if wait_thresh > 5:
                    self.save_and_quit()

            prev_r_l = self.r_l
            prev_r_r = self.r_r

            end = time.time()

            print end - start

if __name__ == "__main__":
    rospy.init_node("recorder_node")
    if len(sys.argv) < 2:
        print "ERROR: Please provide correct arguments"
        sys.exit()
    recording = Recording(sys.argv[1])
    recording.start_recording()