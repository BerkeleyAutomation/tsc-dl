import numpy as np
import matplotlib.pyplot as plt
import IPython

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
# if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print("Downloading pre-trained CaffeNet model...")
#     !../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

path_to_image = "/home/animesh/research/"
image_name = "image-354.jpg"


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0, title="no-title"):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure()
    plt.title(title)
    plt.imshow(data)
    # plt.savefig("feature_extraction/feature_images/" + title + image_name)

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(path_to_image + image_name))
out = net.forward()
# print("Predicted class is #{}.".format(out['prob'].argmax()))

IPython.embed()

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

conv1filters = net.params['conv1'][0].data
vis_square(conv1filters.transpose(0, 2, 3, 1), title = 'conv1_filters')

feat_conv1 = net.blobs['conv1'].data[0]
vis_square(feat_conv1, padval=1, title = "conv1_features")

conv2filters = net.params['conv2'][0].data
vis_square(conv2filters[:48].reshape(48**2, 5, 5), title = "conv2_filters")

feat_conv2 = net.blobs['conv2'].data[0]
vis_square(feat_conv2, padval=1, title = "conv2_features")

feat_conv3 = net.blobs['conv3'].data[0]
vis_square(feat_conv3, padval=0.5, title = "conv3_features")

feat_conv4 = net.blobs['conv4'].data[0]
vis_square(feat_conv4, padval=0.5, title = "conv4_features")

feat_conv5 = net.blobs['conv5'].data[0]
vis_square(feat_conv5, padval=0.5, title = "conv5_features")

feat_pool5 = net.blobs['pool5'].data[0]
vis_square(feat_pool5, padval=1, title = "pool5_features")

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
# sort top k predictions from softmax output
#top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#print labels[top_k]

IPython.embed()

