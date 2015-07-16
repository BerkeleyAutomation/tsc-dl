import matlab.engine as mateng
from time import time
import IPython
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda, random_projection, preprocessing, covariance)
from scipy import linalg
import argparse
import pickle
import cv2
import utils

def parse():
	eng = mateng.start_matlab()

	[s, X] = eng.read_binary_blob(PATH_TO_DATA + '1.conv5b', nargout = 2)
	i = 17;
	while i <= 1489:
		[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(i) + '.conv5b', nargout = 2)
		data = np.array(data)
		X = np.concatenate((X, data), axis = 0)
		i += 16
	return X

def parse_annotations_pickle(annotations_file, PATH_TO_DATA, layer):
	index_map = {}
	label_map = {}
	eng = mateng.start_matlab()
	map_index_data = pickle.load(open(annotations_file, "rb"))
	X = None
	i = 0
	for index in map_index_data:
		print "Parsing label " + str(index) 
		segments = map_index_data[index]
		for seg in segments:
			j = seg[0]
			while j <= seg[1]:
				print j
				if X is None:
					[s, X] = eng.read_binary_blob(PATH_TO_DATA + str(j) + '.' + args.layer, nargout = 2)
				else:
					[s, data] = eng.read_binary_blob(PATH_TO_DATA + str(j) + '.' + args.layer, nargout = 2)
					data = np.array(data)
					X = np.concatenate((X, data), axis = 0)
				index_map[i] = j
				label_map[i] = index
				j += 16
				i += 1
	return X, label_map, index_map

def plot_all(X):
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
	#----------------------------------------------------------------------
	# Pre-processing
	print "t-SNE Scaling"
	X_scaled = preprocessing.scale(X) #zero mean, unit variance
	X_tsne_scaled = tsne.fit_transform(X_scaled)

	#normalize the data (scaling individual samples to have unit norm)
	print "t-SNE L2 Norm"
	X_normalized = preprocessing.normalize(X, norm='l2')
	X_tsne_norm = tsne.fit_transform(X_normalized)


	#whiten the data 
	print "t-SNE Whitening"
	# the mean computed by the scaler is for the feature dimension. 
	# We want the normalization to be in feature dimention. 
	# Zero mean for each sample assumes stationarity which is not necessarily true for CNN features.
	# X: NxD where N is number of examples and D is number of features. 

	# scaler = preprocessing.StandardScaler(with_std=False).fit(X)
	scaler = preprocessing.StandardScaler().fit(X) #this scales each feature to have std-dev 1
	X_centered = scaler.transform(X)

	# U, s, Vh = linalg.svd(X_centered)
	shapeX = X_centered.shape
	IPython.embed()
	# this is DxD matrix where D is the feature dimension
	# still to figure out: It seems computation is not a problem but carrying around a 50kx50k matrix is memory killer!
	sig = (1/shapeX[0]) * np.dot(X_centered.T, X_centered)
	sig2= covariance.empirical_covariance(X_centered, assume_centered=True) #estimated -- this is better.
	sig3, shrinkage= covariance.oas(X_centered, assume_centered=True) #estimated 

	U, s, Vh = linalg.svd(sig, full_matrices=False)
	eps = 1e-2 # this affects how many low- freq eigevalues are eliminated
	invS = np.diag (np.reciprocal(np.sqrt(s+eps)))

	#PCA_whiten
	X_pca = np.dot(invS, np.dot(U.T, X_centered))
	X_tsne_pca = tsne.fit_transform(X_pca)

	#whiten the data (ZCA)
	X_zca = np.dot(U, X_pca)
	X_tsne_zca = tsne.fit_transform(X_zca)

	return X_tsne_scaled, X_tsne_norm, X_tsne_pca, X_tsne_zca

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("file_name", help = "Please specify MAIN file name")
	parser.add_argument("layer", help = "Please specify layer")
	parser.add_argument("PATH_TO_DATA", help="Please specify the path to the feature data")
	parser.add_argument("--a", help = "Annotated frames")
	parser.add_argument("--PATH_TO_DATA_2", help="Please specify the path to 2nd set of feature data")
	parser.add_argument("--a_2", help="Annotated frames for 2nd set of data")
	parser.add_argument("--image", help="Parse image mode", default = None)
	args = parser.parse_args()
	if args.a_2 and args.PATH_TO_DATA_2 and not args.image:
		X1, label_map_1, index_map_1 = parse_annotations_pickle(args.a, args.PATH_TO_DATA, args.layer)
		X2, label_map_2, index_map_2 = parse_annotations_pickle(args.a_2, args.PATH_TO_DATA_2, args.layer)
		X1_pca = utils.pca(X1)
		X2_pca = utils.pca(X2)
		plot_annotated_joint(X1_pca, X2_pca, label_map_1, index_map_1, label_map_2, index_map_2, figure_name = args.file_name +".png", title = "PCA " + args.layer)
	elif args.image and not args.PATH_TO_DATA_2:
			X, label_map, index_map  = utils.parse_annotations_images(args.a, args.PATH_TO_DATA)
			pickle.dump(X, open(args.file_name + "_allimages.p", "wb"))
			pickle.dump(label_map, open(args.file_name + "_labelmap.p", "wb"))
			pickle.dump(index_map, open(args.file_name + "_indexmap.p", "wb"))
			IPython.embed()
			X_pca = utils.pca(X)
			X_tsne = utils.tsne(X)
			X_tsne_pca = utils.tsne_pca(X)
			utils.plot_annotated_embedding(X_pca, label_map, index_map, args.file_name + '_' + args.layer + '_pca.png', 'PCA ' + args.layer)
			utils.plot_annotated_embedding(X_tsne, label_map, index_map, args.file_name + '_' + args.layer + '_tsne.png', 't-SNE ' + args.layer)
			utils.plot_annotated_embedding(X_tsne_pca, label_map, index_map, args.file_name + '_' + args.layer + '_tsne_pca.png', 't-SNE (PCA Input) ' + args.layer)
	else:
		if args.a:
			X, label_map, index_map  = parse_annotations_pickle(args.a, args.PATH_TO_DATA, args.layer)
		else:
			X, label_map, index_map = parse_annotations(args)

		X_pca = utils.pca(X)
		X_tsne = utils.tsne(X)
		X_tsne_pca = utils.tsne_pca(X)
		utils.plot_annotated_embedding(X_pca, label_map, index_map, args.file_name + '_' + args.layer + '_pca.png', 'PCA - C3D ' + args.layer)
		utils.plot_annotated_embedding(X_tsne, label_map, index_map, args.file_name + '_' + args.layer + '_tsne.png', 't-SNE - C3D ' + args.layer)
		utils.plot_annotated_embedding(X_tsne_pca, label_map, index_map, args.file_name + '_' + args.layer + '_tsne_pca.png', 't-SNE(PCA input) - C3D ' + args.layer)