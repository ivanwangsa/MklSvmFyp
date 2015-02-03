'''
Created on 2 Feb, 2015

@author: ivanwangsa
'''

import mklsvmfyp.classifier as clsf
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import sklearn.svm

N = 100
X = np.zeros((N, 2))
for i in range(N):
    X[i, 0] = np.random.random()* 2 - 1
    X[i, 1] = np.random.random() * 2 - 1

y = np.zeros(N)

for i in range(N):
    y[i] = np.sign(X[i, 0] - X[i, 1] + 10 * (np.sin(X[i, 1]) - np.sin(X[i, 0])))
    if np.random.random() > 0.8:
        y[i] = -y[i]


print 'dataset_initialized'

svm1 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(.3), constraint = .3)
svm1.fit(X, y)
svm1.visualize_2d(size = (50, 50))

svm3 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(.3), constraint = 3.)
svm3.fit(X, y)
svm3.visualize_2d(size = (50, 50))

svm2 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(1.), constraint = .3)
svm2.fit(X, y)
svm2.visualize_2d(size = (50, 50))

svm4 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(1.), constraint = 3.)
svm4.fit(X, y)
svm4.visualize_2d(size = (50, 50))

plt.show()