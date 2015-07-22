import IPython
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
	random_projection, preprocessing, covariance, cluster, neighbors)
import utils

def encode_cluster_normalize(X):
	return encode_VLAD(X, 1, 5)

def difference_vectors(X, cluster_predictions, clusters):
	PC = X.shape[1]
	X_vlad =  (X[0] - clusters[cluster_predictions[0]]).reshape(1, PC)
	num_frms = X.shape[0]
	for frm in range(1, num_frms):
		X_diff_frm = (X[frm] - clusters[cluster_predictions[frm]]).reshape(1, PC)
		X_vlad = np.concatenate((X_vlad, X_diff_frm), axis = 0)
	assert X_vlad.shape[0] == X.shape[0]
	assert X_vlad.shape[1] == X.shape[1]
	return X_vlad

def encode_VLAD(X, K, PC):
	IPython.embed()
	X_pca = pca(X, PC = PC)

	kmeans = cluster.KMeans(init = 'k-means++', n_clusters = K)
	kmeans.fit(X_pca)
	clusters = kmeans.cluster_centers_
	assert len(clusters) == K

	neigh = neighbors.KNeighborsClassifier(n_neighbors = 1)
	Y = np.arange(K)
	neigh.fit(clusters, Y)
	cluster_predictions = neigh.predict(X_pca)
	assert len(cluster_predictions) == X.shape[0]

	X_vlad = difference_vectors(X_pca, cluster_predictions, clusters)
	return X_vlad

