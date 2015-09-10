import numpy as np
import yaml

def parse_yaml(yaml_fname):
	config = yaml.load(open(yaml_fname, 'r'))
	return config

config = parse_yaml("../config/needle_passing.yaml")

CAFFE_ROOT = '/home/animesh/caffe/'

TASK_NAME = config['TASK_NAME']

PATH_TO_DATA = config['PATH_TO_DATA']

PATH_TO_CLUSTERING_RESULTS = config['PATH_TO_CLUSTERING_RESULTS']

PATH_TO_KINEMATICS = config['PATH_TO_KINEMATICS']

PATH_TO_OPENCV_2_4_9 = "~/opencv_2.4.9/opencv-2.4.9/lib/"

# Old Color Map with default values from Mathplotlib
# color_map = {1:'b', 2:'g', 3:'r', 4:'c', 5: 'm', 6:'y', 7:'k', 8:'#4B0082', 9: '#9932CC', 10: '#E9967A', 11: '#800000', 12: '#008080'}

# Nicer color map
# color_map = {1:'#00b5e2', 2:'#B9D3B6', 3:'#CFDD45', 4:'#859438', 5: '#e04e39', 6:'#00A598', 7:'k', 8:'#ffc72c', 9: '#9932CC', 10: '#E9967A', 11: '#584F29', 12: '#008080'}

# Even Nicer color map
if TASK_NAME in ["100", "010", "011", "plane", "lego", "people"]:
	color_map = {1:'#00A598', 2:'#ffc72c', 3:'#e04e39', 4:'#00b5e2', 5: '#B9D3B6', 6:'#00b2a9', 7:'k', 8:'#ffc72c', 9: '#9932CC', 10: '#E9967A', 11: '#584F29', 12: '#008080'}
else:
	color_map = {1:'#00A598', 2:'#00b2a9', 3:'#e04e39', 4:'#ffc72c', 5: '#B9D3B6', 6:'#00b5e2', 7:'k', 8:'#ffc72c', 9: '#9932CC', 10: '#E9967A', 11: '#584F29', 12: '#008080'}

alphabet_map = {}
for i in range(500):
	alphabet_map[i] = "A" + str(i) + "_"

# alphabet_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9:"I", 10: "J",
# 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T", 21:"U", 22:"V", 23:"W",
# 24: "X", 25:"Y", 26:"Z", 27:"A1", 28:"A2", 29:"A3", 30:"A4", 31:"A5", 32:"A6", 33:"A7", 34:"A8", 35:"A9", 36:"A10",
# 37:"A11", 38:"A12", 39:"A13", 40:"A14", 41:"A15", 42:"A16", 43:"A17", 44:"A18", 45:"A19", 46:"A20", 47:"A21", 48:"A22", 49: "A23", 50:"A24"}

# Extra colors added:
# E9967A is beige/dark salmon
# 4B0082 is Indigo
# 800000 is Maroon 
# 008080 IS Teal

alex_net_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7']

vgg_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3','conv5_1', 'conv5_2', 'conv5_3', 'pool5']

PATH_TO_SAVE_FIG = '/home/animesh/DeepMilestones/plots/'

# constants.VGG_MEAN = np.array([123.68, 116.779, 103.939])   RGB 0-255 scale
VGG_MEAN = np.array([ 0.48501961,  0.45795686,  0.40760392])

NET_PARAMS = {"AlexNet": [CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt', CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
alex_net_layers], "VGG_SOS": [CAFFE_ROOT + 'models/vgg_sos/deploy.prototxt', CAFFE_ROOT + 'models/vgg_sos/VGG16_SalObjSub.caffemodel', vgg_layers],
"VGG": [CAFFE_ROOT + 'models/vgg/deploy.prototxt', CAFFE_ROOT + 'models/vgg/VGG_ILSVRC_16_layers.caffemodel', vgg_layers]}

CONFIG_FILE = config['CONFIG_FILE']

VIDEO_FOLDER = "video/"

NEW_FRAMES_FOLDER = "frames/"

NEW_BGSUB_FOLDER = "bgsubframes/"

TRANSCRIPTIONS_FOLDER = "transcriptions/"

ANNOTATIONS_FOLDER = "annotations/"

ALEXNET_FEATURES_FOLDER = config["ALEXNET_FEATURES_FOLDER"]

VGG_FEATURES_FOLDER = config["VGG_FEATURES_FOLDER"]

SIFT_FEATURES_FOLDER = config["SIFT_FEATURES_FOLDER"]

PROC_FEATURES_FOLDER = config["PROC_FEATURES_FOLDER"]

CROP_PARAMS_CAPTURE_1 = config["CROP_PARAMS_CAPTURE_1"]

CROP_PARAMS_CAPTURE_2 = config["CROP_PARAMS_CAPTURE_2"]

CROP_PARAMS = {"capture1": CROP_PARAMS_CAPTURE_1, "capture2": CROP_PARAMS_CAPTURE_2}

SR = config["SR"]

CAMERA = config["CAMERA"]

PRUNING_FACTOR_W = config["PRUNING_FACTOR_W"]

PRUNING_FACTOR_Z = config["PRUNING_FACTOR_Z"]

PRUNING_FACTOR_ZW = config["PRUNING_FACTOR_ZW"]

REMOTE = config["REMOTE"]

SIMULATION = config["SIMULATION"]

KINEMATICS_DIM = config["KINEMATICS_DIM"]

N_COMPONENTS_CP = config["N_COMPONENTS_CP"]

N_COMPONENTS_L1 = config["N_COMPONENTS_L1"]

N_COMPONENTS_L2 = config["N_COMPONENTS_L2"]

N_COMPONENTS_CP_W = config["N_COMPONENTS_CP_W"]

N_COMPONENTS_L1_W = config["N_COMPONENTS_L1_W"]

N_COMPONENTS_CP_Z = config["N_COMPONENTS_CP_Z"]

N_COMPONENTS_L1_Z = config["N_COMPONENTS_L1_Z"]

N_COMPONENTS_TIME_W = config["N_COMPONENTS_TIME_W"]

N_COMPONENTS_TIME_Z = config["N_COMPONENTS_TIME_Z"]

N_COMPONENTS_TIME_ZW = config["N_COMPONENTS_TIME_ZW"]

TEMPORAL_WINDOW_ZW = config["TEMPORAL_WINDOW_ZW"]

TEMPORAL_WINDOW_W = config["TEMPORAL_WINDOW_W"]

TEMPORAL_WINDOW_Z = config["TEMPORAL_WINDOW_Z"]

ALPHA_W_CP = config["ALPHA_W_CP"]

ALPHA_Z_CP = config["ALPHA_Z_CP"]

ALPHA_ZW_CP = config["ALPHA_ZW_CP"]

DPGMM_DIVISOR = config["DPGMM_DIVISOR"]

DPGMM_DIVISOR_L1 = config["DPGMM_DIVISOR_L1"]

map_surgeme_label = {'G1': 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6, "G7": 7, "G8": 8, "G9": 9,
"G10": 10, "G12": 12, "G11": 11, "G13": 13, "G14": 14, "G15": 15, "G16": 16, "G17": 17}

caffe_conv_dimensions = {'conv3': (384, 13), 'conv4':(384, 13), 'pool5': (256, 6), 'conv5_3': (512, 14), 'conv5_1': (512, 14)}

train_test_ratio = 0.3