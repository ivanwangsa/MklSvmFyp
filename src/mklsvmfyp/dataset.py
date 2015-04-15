import numpy as np
import os
import csv
from sklearn.datasets.samples_generator import make_blobs, make_classification


class DataSet:
    
    @staticmethod
    def partition_set(self, dataset, proportion = (2., 0., 1.)):
        X = dataset[0]
        y = dataset[1]
        N = X.shape[0]
        partitioned_dataset = []
        proportion = np.array(proportion) / sum(proportion) * N
        num_prop = len(proportion)
        indices = np.hstack(([0],np.ceil(np.cumsum(proportion))))
        for i in range(num_prop):
            partitioned_dataset.append((X[indices[i]:indices[i+1]], y[indices[i], indices[i+1]]))
        return tuple(partitioned_dataset)
    
    @staticmethod
    def normalize_feature_space(self, X, vec = None):
        m = X.shape[1]
        if vec == None:
            res = []
            for i in xrange(m):
                mean = np.mean(X[:,i])
                std = np.std(X[:,i])
                X[:,i] = (X[:,i] - mean)/std
                res.append((mean, std))
            return res
        elif len(vec) == m:
            for i in range(m):
                mean, std = vec[i]
                X[:,i] = (X[:,i] - mean)/std
            return vec
        else:
            return None
    
    def retrieve_sets(self, random_seed = 0):
        np.random.seed(random_seed)
        indices = range(self._num_of_obs)
        np.random.shuffle(indices)
        indices = np.array(indices)
        return self._X[indices], self._y[indices]
        
    def load_ionosphere(self):
        with open('dataset/ionosphere.data') as csvfile:
            data_reader = csv.reader(csvfile, delimiter = ',')
            self._num_of_feat = 34
            self._num_of_obs = 351
            X = np.zeros((self._num_of_obs, self._num_of_feat))
            y = np.zeros(self._num_of_obs)
            i = 0
            for row in data_reader:
                X[i] = row[0: self._num_of_feat]
                y[i] = 1. if row[self._num_of_feat] == 'g' else -1.
                i += 1
            self._X = X
            self._y = y
            
    def generate_blob(self, n_samples=100, n_features=2, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None):
        self._num_of_obs = n_samples
        self._num_of_feat = n_features
        self._X, self._y = make_blobs(n_samples, n_features, centers, cluster_std, center_box, shuffle, random_state)
    
    def generate_classification(self, n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None):
        self._num_of_obs = n_samples
        self._num_of_feat = n_features
        self._X, self._y = make_classification(n_samples, n_features, n_informative, n_redundant, n_repeated, n_classes, n_clusters_per_class, weights, flip_y, class_sep, hypercube, shift, scale, shuffle, random_state)
    
    def save_data(self, name):
        with open(os.path.join('dataset/', name), 'w') as csvfile:
            data_writer = csv.writer(csvfile, delimiter=',')
            rows = np.hstack((self._X, np.reshape(self._y, (self._num_of_obs, 1))))
            data_writer.writerows(rows)
    
    def load_data(self, name):
        with open(os.path.join('dataset/', name)) as csvfile:
            data_reader = csv.reader(csvfile, delimiter = ',')
            self._X = list()
            self._y = list()
            for row in data_reader:
                x = list()
                for col in range(len(row)):
                    if col < len(row) - 1:
                        x.append(float(row[col]))
                    else:
                        self._y.append(float(row[col]))
                self._X.append(np.array(x))
            self._X = np.array(self._X)
            self._y = np.array(self._y)
            self._num_of_obs, self._num_of_feat = self._X.shape
        
    def __init__(self):
        pass