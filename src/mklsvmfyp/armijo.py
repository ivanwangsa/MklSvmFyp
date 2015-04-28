# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 08:29:37 2015

@author: ivanwangsa
"""

from dataset import DataSet
from classifier import PriorMklSvm, Kernel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = DataSet()
    data.generate_classification(n_samples = 400, n_features=15, n_informative=5, n_redundant=7, random_state = 10)
    blob2_data = data.retrieve_sets()
    list_kernels = list()
    for sigma in [0.01, 0.03, 0.1, 0.3, 0.7, 1., 2., 3., 5., 8., 10.]:
        list_kernels.append(Kernel.gaussian(sigma))
    
    beta = 0.5
    sigma = 0.5    
    
    priorMkl = PriorMklSvm(constraint = 1., kernels = list_kernels)
    priorMkl.set_armijo_beta(beta)
    priorMkl.set_armijo_sigma(sigma)
    priorMkl.fit(blob2_data[0], blob2_data[1])
    armijo_true = priorMkl._steps
    
    priorMkl = PriorMklSvm(constraint = 1., kernels = list_kernels, armijo_modified = False)
    priorMkl.set_armijo_beta(beta)
    priorMkl.set_armijo_sigma(sigma)    
    priorMkl.fit(blob2_data[0], blob2_data[1])
    armijo_false = priorMkl._steps

    fig = plt.figure()    
    ax = plt.gca()
    ax.set_yscale('log')
    
    plt.plot(range(1, len(armijo_true) + 1), armijo_true, 'k', linewidth = 2.0, label = 'Modified Armijo, Projected')
    plt.plot(range(1, len(armijo_false) + 1), armijo_false, 'k--', linewidth = 2.0, label = 'Original Armijo, Projected')    
    
    priorMkl = PriorMklSvm(constraint = 1., kernels = list_kernels, method='conditional')
    priorMkl.set_armijo_beta(beta)
    priorMkl.set_armijo_sigma(sigma)    
    priorMkl.fit(blob2_data[0], blob2_data[1])
    print priorMkl.kernel_coefficients
    armijo_true = priorMkl._steps
    
    priorMkl = PriorMklSvm(constraint = 1., kernels = list_kernels, armijo_modified = False, method='conditional')
    priorMkl.set_armijo_beta(beta)
    priorMkl.set_armijo_sigma(sigma)    
    priorMkl.fit(blob2_data[0], blob2_data[1])
    armijo_false = priorMkl._steps
    
    plt.plot(range(1, len(armijo_true) + 1), armijo_true, 'b', linewidth = 2.0, label = 'Modified Armijo, Conditional')
    plt.plot(range(1, len(armijo_false) + 1), armijo_false, 'b--', linewidth = 2.0, label = 'Original Armijo, Conditional')    
    
    plt.xlabel('Number of iteration')
    plt.ylabel('Step sizes')
    plt.legend(loc = 'best', prop={'size': 12})
    plt.title('beta = 0.8, sigma = 0.2')
    plt.show()