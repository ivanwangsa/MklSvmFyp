'''
Created on 2 Feb, 2015

@author: ivanwangsa
'''
import csv
import numpy as np
from mklsvmfyp.classifier import Kernel, SilpMklSvm, SoftMargin1Svm
from sklearn.svm.classes import SVC
import os

class DataSet:
    
    def retrieve_sets(self, random_seed = 0, proportion = (2., 0., 1.)):
        np.random.seed(random_seed)
        indices = np.arange(self._num_of_obs)
        np.random.shuffle(indices)
        proportion = (proportion[0]/sum(proportion), proportion[1]/sum(proportion), proportion[2]/sum(proportion))
        start_train = 0
        start_valid = int(self._num_of_obs * proportion[0])
        start_test  = int(self._num_of_obs * (proportion[0] + proportion[1]))
        end = self._num_of_obs
        
        training_set_indices = indices[start_train : start_valid]
        validation_set_indices = indices[start_valid : start_test]
        test_set_indices = indices[start_test : end]
        
        training_set = (self._X[training_set_indices], self._y[training_set_indices])
        validation_set = (self._X[validation_set_indices], self._y[validation_set_indices])
        test_set = (self._X[test_set_indices], self._y[test_set_indices])
        
        return (training_set, validation_set, test_set)
        
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

