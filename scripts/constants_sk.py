import numpy as np

CAFFE_ROOT = '/home/animesh/caffe/'

PATH_TO_DATA = "/home/animesh/DeepMilestones/jigsaws/Suturing_video/"

PATH_TO_CLUSTERING_RESULTS = "/home/animesh/DeepMilestones/clustering_FCED/"

PATH_TO_KINEMATICS = "/home/animesh/DeepMilestones/jigsaws/Suturing_kinematics/kinematics/AllGestures/"

color_map = {1:'b', 2:'g', 3:'r', 4:'c', 5: 'm', 6:'y', 7:'k', 8:'#4B0082', 9: '#9932CC', 10: '#E9967A', 11: '#800000', 12: '#008080'}

alphabet_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9:"I", 10: "J",
11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P"}

# Extra colors added:
# E9967A is beige/dark salmon
# 4B0082 is Indigo
# 800000 is Maroon 
# 008080 IS Teal

alex_net_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7']

vgg_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3','conv5_1', 'conv5_2', 'conv5_3', 'pool5']

PATH_TO_SAVE_FIG = '/home/animesh/DeepMilestones/plots_sk/'

# constants.VGG_MEAN = np.array([123.68, 116.779, 103.939])   RGB 0-255 scale
VGG_MEAN = np.array([ 0.48501961,  0.45795686,  0.40760392])

NET_PARAMS = {"AlexNet": [CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt', CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
alex_net_layers], "VGG_SOS": [CAFFE_ROOT + 'models/vgg_sos/deploy.prototxt', CAFFE_ROOT + 'models/vgg_sos/VGG16_SalObjSub.caffemodel', vgg_layers],
"VGG": [CAFFE_ROOT + 'models/vgg/deploy.prototxt', CAFFE_ROOT + 'models/vgg/VGG_ILSVRC_16_layers.caffemodel', vgg_layers]}

CONFIG_FILE = "meta_file_Suturing.txt"

VIDEO_FOLDER = "video/"

NEW_FRAMES_FOLDER = "frames/"

NEW_BGSUB_FOLDER = "bgsubframes/"

TRANSCRIPTIONS_FOLDER = "transcriptions/"

ANNOTATIONS_FOLDER = "annotations/"

ALEXNET_FEATURES_FOLDER = "alexnetfeatures_2/"

VGG_FEATURES_FOLDER = "vggfeatures_2/"

PROC_FEATURES_FOLDER = "features_FCED/"

CROP_PARAMS = {"capture2": "\"crop=330:260:150:150\"", "capture1": "\"crop=330:260:200:170\""}

map_surgeme_label = {'G1': 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5, "G6": 6, "G7": 7, "G8": 8, "G9": 9,
"G10": 10, "G12": 12, "G11": 11, "G13": 13, "G14": 14, "G15": 15, "G16": 16, "G17": 17}

train_test_ratio = 0.3
