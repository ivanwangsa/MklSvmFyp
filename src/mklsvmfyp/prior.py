'''
Created on 13 Apr, 2015

@author: ivanwangsa
'''

from dataset import DataSet
from classifier import PriorMklSvm, Kernel
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from copy import deepcopy

if __name__ == '__main__':
    datasets = DataSet()
    datasets.load_ionosphere()
    data = datasets.retrieve_sets()
    length = len(data[1])
    part = int(0.6 * length)
    train = data[0][:part], data[1][:part]
    test = data[0][part:], data[1][part:]
    list_kernels = list()
    delta = []
    params = [0.01, 0.03, 0.1, 0.3, 0.7, 1., 2., 3., 5., 8., 10.]
    for sigma in params:
        list_kernels.append(Kernel.gaussian(sigma))
        delta.append(np.exp(-np.abs(np.log(sigma))/7))
    delta = np.array(delta) * 11./np.sum(delta)
    plt.plot(params, delta, 'og-', label = 'distinct delta')
    plt.plot(params, np.repeat(1., 11), 'ob-', label = 'delta = ones')    
        
    # plt.yscale('log')    
    # plt.xscale('log')
    regularization_parameters = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30., 100.]    
   
    max_C = 0
    max_score = 0
    max_std = 0
    max_priorMKl = None
    for C in regularization_parameters:
        priorMklSvm = PriorMklSvm(constraint = C, delta = delta, kernels = list_kernels, method = 'projected', armijo_modified=True)
        priorMklSvm.set_armijo_beta(0.5)
        priorMklSvm.set_armijo_sigma(0.5)
        priorMklSvm.fit(train[0], train[1])
        score = priorMklSvm.cross_validate(K = 10)
        score_mean = np.mean(score)
        score_std = np.std(score)
        if max_score < score_mean or (max_score < score_mean + 1e-7 and max_std > score_std):
            max_score = score_mean
            max_std = score_std
            max_C = C
            max_priorMKl = deepcopy(priorMklSvm)
     
    print max_C, max_priorMKl.score(train[0], train[1]), max_score, max_std, max_priorMKl.score(test[0], test[1])
    print max_priorMKl.kernel_coefficients
    
    
    max_C = 0
    max_score = 0
    max_std = 0
    max_priorMKl = None
    for C in regularization_parameters:
        priorMklSvm = PriorMklSvm(constraint = C, kernels = list_kernels, method = 'projected', armijo_modified=True)
        priorMklSvm.set_armijo_beta(0.5)
        priorMklSvm.set_armijo_sigma(0.5)
        priorMklSvm.fit(train[0], train[1])
        score = priorMklSvm.cross_validate(K = 10)
        score_mean = np.mean(score)
        score_std = np.std(score)
        if max_score < score_mean or (max_score < score_mean + 1e-7 and max_std > score_std):
            max_score = score_mean
            max_std = score_std
            max_C = C
            max_priorMKl = deepcopy(priorMklSvm)
     
    print max_C, max_priorMKl.score(train[0], train[1]), max_score, max_std, max_priorMKl.score(test[0], test[1])
    print max_priorMKl.kernel_coefficients
    
    
    