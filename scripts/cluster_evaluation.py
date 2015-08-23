import numpy as np
from sklearn import mixture
import IPython
import utils
import scipy.spatial.distance as distance

def split(points, predictions):
	points_list = {}
	for i in range(len(predictions)):
		label = predictions[i]
		if label not in points_list:
			points_list[label] = utils.reshape(points[i])
		else:
			curr = points_list[label]
			curr = np.concatenate((curr, utils.reshape(points[i])), axis = 0)
			points_list[label] = curr
	return points_list

def dunn_index(points, predictions, means):
	if len(points) == 0:
		return [None, None, None]
	points_in_clusters = split(points, predictions)	
	delta_list_1 = []
	delta_list_2 = []
	delta_list_3 = []

	# Wikipedia Definition No. 1 for Delta - maximum distance between all point-pairs in cluster
	for cluster in points_in_clusters.keys():
		if len(points_in_clusters[cluster]) > 1:
			try:
				delta_list_1.append(max(distance.pdist(points_in_clusters[cluster], 'euclidean')))
			except ValueError as e:
				print e
				IPython.embed()

	# Wikipedia Definition No. 2 for Delta - mean distance between all point-pairs in cluster
	for cluster in points_in_clusters.keys():
		if len(points_in_clusters[cluster]) > 1:
			delta_list_2.append(np.mean(distance.pdist(points_in_clusters[cluster], 'euclidean')))

	# Wikipedia Definition No. 3 for Delta - distance of all points from mean
	for cluster in points_in_clusters.keys():
		if len(points_in_clusters[cluster]) > 1:
			delta_list_3.append(np.mean(distance.cdist(points_in_clusters[cluster], utils.reshape(means[cluster]), 'euclidean')))

	del_list = distance.pdist(means, 'euclidean')

	try:
		dunn_index_1 = min(del_list) / max(delta_list_1)
		dunn_index_2 = min(del_list) / max(delta_list_2)
		dunn_index_3 = min(del_list) / max(delta_list_3)
	except ValueError as e:
		print e
		return [None, None, None]

	return [dunn_index_1, dunn_index_2, dunn_index_3]

if __name__ == "__main__":
	points = np.arange(4000).reshape(10, 400)

	g = mixture.GMM(n_components=4)
	g.fit(points)

	predictions = g.predict(points)
	means = g.means_


