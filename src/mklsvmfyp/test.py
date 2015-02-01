'''
Created on 2 Feb, 2015

@author: ivanwangsa
'''

import mklsvmfyp.classifier as clsf
import numpy as np

N = 100
X = np.zeros((N, 2))
for i in range(N):
    X[i, 0] = np.random.random()* 2 - 1
    X[i, 1] = np.random.random() * 2 - 1

y = np.zeros(N)

for i in range(N):
    y[i] = np.sign(X[i, 0] - X[i, 1] + 10 * (np.sin(X[i, 1]) - np.sin(X[i, 0])))

svm = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(0.2), constraint = 1.)
svm.fit(X, y)
svm.visualize_2d(size = (50, 50))

nn5 = clsf.NearestNeighbour(number_of_neighbors=5)
nn5.fit(X,y)
nn5.visualize_2d(size = (50, 50))