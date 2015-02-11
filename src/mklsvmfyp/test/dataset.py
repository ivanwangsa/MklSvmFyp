'''
Created on 2 Feb, 2015

@author: ivanwangsa
'''
import csv
import numpy as np
from mklsvmfyp.classifier import Kernel, SilpMklSvm, SoftMargin1Svm
from sklearn.svm.classes import SVC
import os
from numpy import std

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
        indices = np.arange(self._num_of_obs)
        np.random.shuffle(indices)
        
        return self._X[indices], self._y[indices]
        
    def _load_ionosphere(self):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'ionosphere.data')) as csvfile:
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
            
    def __init__(self, dataset = 'ionosphere'):
        if(dataset == 'ionosphere'):
            self._load_ionosphere()

