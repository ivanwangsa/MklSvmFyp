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
    plt.clf()
    datasets = DataSet()
    datasets.load_ionosphere()
    data = datasets.retrieve_sets()
    length = len(data[1])
    part = int(0.6 * length)
    train = data[0][:part], data[1][:part]
    test = data[0][part:], data[1][part:]
    list_kernels = list()
    delta = []
    params = [0.1, 0.2, 0.3, 0.5, 1., 2., 3., 5., 10.]
    for sigma in params:
        list_kernels.append(Kernel.gaussian(sigma))
    start_color = 16711680
    end_color = 255
    zetas = [3., 4., 6., 10., 20., 100., None]
    num_iter = 0
    regularization_parameters = [0.1, 0.3, 1., 3., 10.]    

    priorMklSvm = PriorMklSvm(constraint = 1., delta = None, kernels = list_kernels, method = 'projected', armijo_modified=True)
    priorMklSvm._X = train[0]
    priorMklSvm._y = train[1]
    priorMklSvm._compute_gram_matrices()    
    Cs = []
    trains = []
    mean_CVs = []
    std_CVs = []
    tests = []
    for zeta in zetas:
        increment_color = (end_color - start_color) / (len(zetas) - 1)
        if zeta != None:        
            for sigma in params:
                delta.append(np.exp(-np.abs(np.log(sigma))/zeta))
            delta = np.sum(1/np.array(delta))/len(params) * np.array(delta)
        else:
            delta = np.repeat(1., len(params))
            
        priorMklSvm.set_delta(delta)
        color_string = hex(start_color + increment_color * num_iter)
        num_iter += 1
        color_string = color_string[2:]
        while len(color_string) < 6:
            color_string = '0' + color_string
        color_string = '#' + color_string
        print zeta,
        max_C = 0
        max_score = 0
        max_std = 0
        max_priorMKl = None
        for C in regularization_parameters:
            print C,
            priorMklSvm.set_constraint(C)
            priorMklSvm.fit(train[0], train[1], compute=False)
            score = priorMklSvm.cross_validate(K = 10)
            score_mean = np.mean(score)
            score_std = np.std(score)
            if max_score < score_mean or (max_score < score_mean + 1e-7 and max_std > score_std):
                max_score = score_mean
                max_std = score_std
                max_C = C
                max_priorMKl = deepcopy(priorMklSvm)
        Cs.append(max_C)
        trains.append(max_priorMKl.score(train[0], train[1]))
        mean_CVs.append(max_score)
        std_CVs.append(max_std)
        tests.append(max_priorMKl.score(test[0], test[1]))
        print
        print max_priorMKl.kernel_coefficients
        plt.plot(params, max_priorMKl.kernel_coefficients, 'o-', color=color_string, label='zeta = ' + str(zeta))
        delta = []
    plt.xscale('log')
    plt.legend(loc='best', prop={'size': 12})
    plt.show()
    # plt.yscale('log')    
    
    
    
    
    
    
    
    