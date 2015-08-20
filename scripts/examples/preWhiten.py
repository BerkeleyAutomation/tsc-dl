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

# PATH_TO_DATA = 'data/8fps_nooverlap_conv5b_cropped_annotations/'
PATH_TO_DATA = 'data/brownie/'

all_segments = {'1': [(176, 248), (496, 566), (739, 780), (977, 1075), (1176,1236), (1352,1405)], '2': [(585, 640), (796, 830), (1267, 1284)],
'3': [(695, 720), (857, 889), (1126, 1156), (1315, 1333), (1414, 1450)]}

color_map = {1:'b', 2:'g', 3:'r', 4:'c', 5: 'm', 6:'y', 7:'k', 8:'#000000', 9: '#9932CC'}
index_map = {}
label_map = {}

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

def parse_annotations():
        eng = mateng.start_matlab()

        [s, X] = eng.read_binary_blob(PATH_TO_DATA + '176.conv5b', nargout = 2)
        color = 'b'
        i = 0
        for label in all_segments:
                print "Parsing label " + label
                segments = all_segments[label]
                if label == '2':
                        color = 'r'
                elif label == '3':
                        color = 'g'
                else:
                        color = 'b'
                for seg in segments:
                        j = seg[0]
                        while j <= seg[1]:
                                if j == 176:
                                        color_map[i] = color
                                        index_map[i] = j
                                        j += 16
                                        i += 1
                                        continue
                                [s, data] = eng.read_binary_blob(PATH_TO_DATA + str(j) + '.conv5b', nargout = 2)
                                data = np.array(data)
                                X = np.concatenate((X, data), axis = 0)
                                color_map[i] = color
                                index_map[i] = j
                                j += 16
                                i += 1
        return X

def parse_annotations_pickle(annotations):
        eng = mateng.start_matlab()
        map_index_data = pickle.load(open(annotations, "rb"))
        X = None
        i = 0
        for index in map_index_data:
                print "Parsing label " + str(index)
                segments = map_index_data[index]
                color = np.random.rand(3,1)
                for seg in segments:
                        j = seg[0]
                        while j <= seg[1]:
                                if X is None:
                                        [s, X] = eng.read_binary_blob(PATH_TO_DATA + str(j) + '.conv5b', nargout = 2)
                                else:
                                        [s, data] = eng.read_binary_blob(PATH_TO_DATA + str(j) + '.conv5b', nargout = 2)
                                        data = np.array(data)
                                        X = np.concatenate((X, data), axis = 0)
                                index_map[i] = j
                                label_map[i] = index
                                j += 16
                                i += 1
        return X

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, figure_name, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure()
        for i in range(X.shape[0]):
                plt.text(X[i, 0], X[i, 1], str(i*16 + 1), color=plt.cm.Set1((i*16 + 1)/ 100), fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([]), plt.yticks([])
        if title is not None:
                plt.title(title)
        plt.savefig('plots/'+figure_name)

# Scale and visualize the embedding vectors
def plot_annotated_embedding(X, figure_name, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure()
        r_x = r_y = []
        g_x = g_y = []
        b_x =  b_y = []
        for i in range(X.shape[0]):
                frm_num = index_map[i]
                color = color_map[i]
                if color == 'r':
                        plt.text(X[i, 0], X[i, 1], 'o', color=color, fontdict={'weight': 'bold', 'size': 9})
                elif color == 'g':
                        plt.text(X[i, 0], X[i, 1], 'x', color=color, fontdict={'weight': 'bold', 'size': 9})
                else:
                        plt.text(X[i, 0], X[i, 1], '*', color=color, fontdict={'weight': 'bold', 'size': 13})
        #       if color == 'r':
        #               r_x.append(X[i, 0])
        #               r_y.append(X[i, 1])
        #       elif color == 'g':
        #               g_x.append(X[i, 0])
        #               g_y.append(X[i, 1])
        #       else:
        #               b_x.append(X[i, 0])
        #               b_y.append(X[i, 1])
        # plt.plot(r_x, r_y, 'rs')
        # plt.plot(g_x, g_y, 'go')
        # plt.plot(b_x, b_y, 'b^')
        plt.xticks([]), plt.yticks([])
        if title is not None:
                plt.title(title)
        plt.savefig('plots/'+figure_name)

# Scale and visualize the embedding vectors
def plot_annotated_embedding_2(X, figure_name, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure()
        r_x = r_y = []
        g_x = g_y = []
        b_x =  b_y = []
        for i in range(X.shape[0]):
                frm_num = index_map[i]
                plt.text(X[i, 0], X[i, 1], 'x', color=color_map[label_map[i]], fontdict={'weight': 'bold', 'size': 5})
        plt.xticks([]), plt.yticks([])
        if title is not None:
                plt.title(title)
        plt.savefig('plots/'+figure_name)

def pca(X):
        print("Computing PCA embedding")
        scaler = preprocessing.StandardScaler(with_std = False).fit(X)
        X_centered = scaler.transform(X)
        X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X_centered)
        return X_pca

def tsne_pca(X):
        print("Computing PCA -> t-SNE embedding")
        scaler = preprocessing.StandardScaler().fit(X)
        X_centered = scaler.transform(X)
        X_pca = decomposition.TruncatedSVD(n_components=40).fit_transform(X_centered)
        tsne = manifold.TSNE(init = 'pca')
        X_tsne = tsne.fit_transform(X_pca)
        return X_tsne

def tsne(X):
        #----------------------------------------------------------------------
        # t-SNE embedding of the digits dataset
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        return X_tsne

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
        parser.add_argument("--a", help = "Annotated frames")
        args = parser.parse_args()
        if args.a:
                X = parse_annotations_pickle(args.a)
        else:
                X = parse_annotations()

        # parser.add_argument("plot", help = "Choose from pca, tsne, etc.")
        # parser.add_argument("figure_name", help = "Figure name to be saved")
        # plotter_func = locals()[args.plot]
        # plotter_func(args.figure_name, X)

        X_pca = pca(X)
        X_tsne = tsne(X)
        plot_annotated_embedding_2(X = X_pca, figure_name = 'brownie_pca_stdFalse.png', title = 'PCA conv5b')
        plot_annotated_embedding_2(X = X_tsne, figure_name = 'brownie_tsne.png', title = 't-SNE conv5b')

        # X_tsne_scaled, X_tsne_norm, X_tsne_pca, X_tsne_zca  = plot_all(X)
        # plotNames = ["X_tsne_scaled", "X_tsne_norm", "X_tsne_pca", "X_tsne_zca"]

        # for plotName in plotNames:
        #       plot_annotated_embedding_2 (X=eval(plotName), figure_name='brownie_'+plotName, title=plotName+'_conv5b')
-- INSERT --                                                                                                                                                                                  249,1         Bot

