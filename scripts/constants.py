import yaml
import numpy as np
import os
import sys

default_config = {'ALEXNET_FEATURES_FOLDER': 'alexnetfeatures/', 'ALPHA_W_CP': 750, 'ALPHA_ZW_CP': 1000, 'ALPHA_Z_CP': 0.001,
 'CAMERA': 'capture2',  'CONFIG_FILE': 'meta_file_Suturing.txt', 'CROP_PARAMS_CAPTURE_1': 'crop=330:260:200:170',
 'CROP_PARAMS_CAPTURE_2': 'crop=330:260:150:150', 'DPGMM_DIVISOR': 45, 'DPGMM_DIVISOR_L1': 15, 'KINEMATICS_DIM': 38,
 'N_COMPONENTS_CP': 5, 'N_COMPONENTS_CP_W': 5, 'N_COMPONENTS_CP_Z': 15, 'N_COMPONENTS_L1': 20,
 'N_COMPONENTS_L1_W': 15, 'N_COMPONENTS_L1_Z': 15, 'N_COMPONENTS_L2': 2, 'N_COMPONENTS_TIME_W': 40,
 'N_COMPONENTS_TIME_Z': 40, 'N_COMPONENTS_TIME_ZW': 40, 'PATH_TO_CLUSTERING_RESULTS': 'clustering_suturing_test/',
 'PATH_TO_DATA': 'Suturing_video/', 'PATH_TO_KINEMATICS': 'Suturing_kinematics/kinematics/AllGestures/',
 'PROC_FEATURES_FOLDER': 'features_E12345/', 'PRUNING_FACTOR_T': 0.49, 'PRUNING_FACTOR_W': 0.75, 'PRUNING_FACTOR_Z': 0.49,
 'PRUNING_FACTOR_ZW': 0.49, 'REMOTE': 1, 'SIFT_FEATURES_FOLDER': 'siftfeatures/', 'SIMULATION': False,
 'SR': 3, 'TASK_NAME': 'suturing', 'TEMPORAL_WINDOW_W': 2, 'TEMPORAL_WINDOW_Z': 2, 'TEMPORAL_WINDOW_ZW': 2,
 'VGG_FEATURES_FOLDER': 'vggfeatures_2/', 'WEIGHTED_PRUNING_MODE': False, 'WEIGHT_EXPERT': 30, 'WEIGHT_INTERMEDIATE': 2}

def parse_yaml(yaml_fname):
	config = yaml.load(open(yaml_fname, 'r'))
	return config

class Config:
	def __init__(self, yaml_fname):
		self.__vars = parse_yaml("../config/"+yaml_fname)
		self.__yaml_fname = yaml_fname

	def get(self, var_name):
		if var_name in self.__vars:
			return self.__vars[var_name]
		if var_name not in default_config:
			print "ERROR: Incorrect variable name, ", str(var_name)
			sys.exit()
		else:
			print "ERROR: Using default parameter for ", str(var_name)
			return default_config[var_name]

f = open('../config/defaultconfig', 'r+')
config = Config(f.readline().strip('\n'))

# Constants for Path resolution
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
CAFFE_ROOT = '/home/animesh/caffe/'
TASK_NAME = config.get('TASK_NAME')
PATH_TO_DATA = REPO_ROOT + "/data/" + config.get('PATH_TO_DATA')
PATH_TO_CLUSTERING_RESULTS = REPO_ROOT + "/clustering/" + config.get('PATH_TO_CLUSTERING_RESULTS')

if not os.path.exists(PATH_TO_CLUSTERING_RESULTS):
	print PATH_TO_CLUSTERING_RESULTS, " does not exist; new folder created"
	os.mkdir(PATH_TO_CLUSTERING_RESULTS)

PATH_TO_KINEMATICS = REPO_ROOT + "/data/" + config.get('PATH_TO_KINEMATICS')
PATH_TO_OPENCV_2_4_9 = "~/opencv_2.4.9/opencv-2.4.9/lib/"
PATH_TO_SAVE_FIG = REPO_ROOT + "/plots/"
CONFIG_FILE = config.get('CONFIG_FILE')
VIDEO_FOLDER = "video/"
NEW_FRAMES_FOLDER = "frames/"
NEW_BGSUB_FOLDER = "bgsubframes/"
TRANSCRIPTIONS_FOLDER = "transcriptions/"
ANNOTATIONS_FOLDER = "annotations/"
ALEXNET_FEATURES_FOLDER = config.get("ALEXNET_FEATURES_FOLDER")
VGG_FEATURES_FOLDER = config.get("VGG_FEATURES_FOLDER")
SIFT_FEATURES_FOLDER = config.get("SIFT_FEATURES_FOLDER")
PROC_FEATURES_FOLDER = config.get("PROC_FEATURES_FOLDER")
CROP_PARAMS_CAPTURE_1 = config.get("CROP_PARAMS_CAPTURE_1")
CROP_PARAMS_CAPTURE_2 = config.get("CROP_PARAMS_CAPTURE_2")

# Nice color maps
if TASK_NAME in ["100", "010", "011", "plane", "lego", "people", "people2"]:
	color_map = {1:'#00A598', 2:'#ffc72c', 3:'#e04e39', 4:'#00b5e2', 5: '#B9D3B6', 6:'#00b2a9', 7:'k', 8:'#ffc72c', 9: '#9932CC', 10: '#E9967A', 11: '#584F29', 12: '#008080'}
else:
	color_map = {1:'#00A598', 2:'#00b2a9', 3:'#e04e39', 4:'#ffc72c', 5: '#B9D3B6', 6:'#00b5e2', 7:'k', 8:'#ffc72c', 9: '#9932CC', 10: '#E9967A', 11: '#584F29', 12: '#008080'}

# Surgeme representations
alphabet_map = {}

for i in range(500):
	alphabet_map[i] = "label" + str(i)

map_surgeme_label = {'G1': 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6, "G7": 7, "G8": 8, "G9": 9,
"G10": 10, "G12": 12, "G11": 11, "G13": 13, "G14": 14, "G15": 15, "G16": 16, "G17": 17}

# Caffe CNN variables
caffe_conv_dimensions = {'conv3': (384, 13), 'conv4':(384, 13), 'pool5': (256, 6), 'conv5_3': (512, 14), 'conv5_1': (512, 14)}
alex_net_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7']
vgg_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3','conv5_1', 'conv5_2', 'conv5_3', 'pool5']
NET_PARAMS = {"AlexNet": [CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt', CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
alex_net_layers], "VGG_SOS": [CAFFE_ROOT + 'models/vgg_sos/deploy.prototxt', CAFFE_ROOT + 'models/vgg_sos/VGG16_SalObjSub.caffemodel', vgg_layers],
"VGG": [CAFFE_ROOT + 'models/vgg/deploy.prototxt', CAFFE_ROOT + 'models/vgg/VGG_ILSVRC_16_layers.caffemodel', vgg_layers]}

# Misc. Parameters
CROP_PARAMS = {"capture1": CROP_PARAMS_CAPTURE_1, "capture2": CROP_PARAMS_CAPTURE_2}
SR = config.get("SR")
CAMERA = config.get("CAMERA")

# Clustering Parameters
PRUNING_FACTOR_W = config.get("PRUNING_FACTOR_W")
PRUNING_FACTOR_Z = config.get("PRUNING_FACTOR_Z")
PRUNING_FACTOR_ZW = config.get("PRUNING_FACTOR_ZW")
PRUNING_FACTOR_T = config.get("PRUNING_FACTOR_T")
REMOTE = config.get("REMOTE")
SIMULATION = config.get("SIMULATION")
KINEMATICS_DIM = config.get("KINEMATICS_DIM")
N_COMPONENTS_CP = config.get("N_COMPONENTS_CP")
N_COMPONENTS_L1 = config.get("N_COMPONENTS_L1")
N_COMPONENTS_L2 = config.get("N_COMPONENTS_L2")
N_COMPONENTS_CP_W = config.get("N_COMPONENTS_CP_W")
N_COMPONENTS_L1_W = config.get("N_COMPONENTS_L1_W")
N_COMPONENTS_CP_Z = config.get("N_COMPONENTS_CP_Z")
N_COMPONENTS_L1_Z = config.get("N_COMPONENTS_L1_Z")
N_COMPONENTS_TIME_W = config.get("N_COMPONENTS_TIME_W")
N_COMPONENTS_TIME_Z = config.get("N_COMPONENTS_TIME_Z")
N_COMPONENTS_TIME_ZW = config.get("N_COMPONENTS_TIME_ZW")
TEMPORAL_WINDOW_ZW = config.get("TEMPORAL_WINDOW_ZW")
TEMPORAL_WINDOW_W = config.get("TEMPORAL_WINDOW_W")
TEMPORAL_WINDOW_Z = config.get("TEMPORAL_WINDOW_Z")
ALPHA_W_CP = config.get("ALPHA_W_CP")
ALPHA_Z_CP = config.get("ALPHA_Z_CP")
ALPHA_ZW_CP = config.get("ALPHA_ZW_CP")
DPGMM_DIVISOR = config.get("DPGMM_DIVISOR")
DPGMM_DIVISOR_L1 = config.get("DPGMM_DIVISOR_L1")
WEIGHT_EXPERT = config.get("WEIGHT_EXPERT")
WEIGHT_INTERMEDIATE = config.get("WEIGHT_INTERMEDIATE")
WEIGHTED_PRUNING_MODE = config.get("WEIGHTED_PRUNING_MODE")

train_test_ratio = 0.3
