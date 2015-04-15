'''
Created on 2 Feb, 2015

@author: ivanwangsa
'''

import mklsvmfyp.classifier as clsf
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import sklearn.svm
from mklsvmfyp.dataset import DataSet

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

# 
# print 'dataset_initialized'
# 
# svm1 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(.3), constraint = .3)
# svm1.fit(X, y)
# svm1.visualize_2d(size = (50, 50))
# 
# svm3 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(.3), constraint = 3.)
# svm3.fit(X, y)
# svm3.visualize_2d(size = (50, 50))
# 
# svm2 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(1.), constraint = .3)
# svm2.fit(X, y)
# svm2.visualize_2d(size = (50, 50))
# 
# svm4 = clsf.SoftMargin1Svm(kernel=clsf.Kernel.gaussian(1.), constraint = 3.)
# svm4.fit(X, y)
# svm4.visualize_2d(size = (50, 50))
# 
# plt.show()
# 
# M = 5
# 
# delta = np.random.random_sample(M)
# delta = delta * M/np.sum(delta)
# 
# # delta = np.array([2./(m+1) for m in range(M)])
# 
# # delta = np.repeat(1., M)
# 
# def project(vector, delta):
#     if np.dot(delta, vector) <= 1:
#         return vector
#     M = len(delta)
#     old_delta = np.copy(delta)
#     items = [(vector[m], delta[m]) for m in range(M)]
#     items = sorted(items, key = lambda x: x[0]/x[1])
#     items = zip(*list(items))
#     why, delta = np.array(items[0]), np.array(items[1])
#     i = M - 1
#     num = 0.
#     den = 0.
#     t_hat = None
#     while i > 0:
#         num += delta[i]*why[i]
#         den += delta[i]**2
#         t_i = (num - 1.)/den
#         if t_i > why[i-1]/delta[i-1]:
#             t_hat = t_i
#             break
#         i -= 1
#     if i == 0:
#         num += delta[i] * why[i]
#         den += delta[i]**2
#         t_hat = (num - 1.)/den
#     return np.maximum(vector - old_delta * t_hat, np.repeat(0.,M))
# 
# def generate_f(delta, vector):
#     M = len(delta)
#     items = [(vector[m], delta[m]) for m in range(M)]
#     items = sorted(items, key = lambda x: x[0]/x[1])
#     items = zip(*list(items))
#     why, delta = np.array(items[0]), np.array(items[1])
#     def f(t):
#         i = 0
#         while i < M:
#             if t <= why[i]/delta[i]:
#                 return t + 0.5*np.sum((delta[i:] * t - why[i:]) * (delta[i:] * t - why[i:]))
#             i += 1
#         i -= 1
#         return t + 0.5*np.sum((delta[i:] * t - why[i:]) * (delta[i:] * t - why[i:]))
#     return f
# 
# def generate_f_prime(delta, vector):
#     M = len(delta)
#     items = [(vector[m], delta[m]) for m in range(M)]
#     items = sorted(items, key = lambda x: x[0]/x[1])
#     items = zip(*list(items))
#     why, delta = np.array(items[0]), np.array(items[1])
#     def f_prime(t):
#         i = 0
#         while i < M:
#             if t <= why[i]/delta[i]:
#                 return 1 + np.sum(delta[i:] * (delta[i:] * t - why[i:]))
#             i += 1
#         i -= 1
#         return 1 + np.sum(delta[i:]  * (delta[i:] * t - why[i:]))
#     return f_prime
# 
# def bin_search(fn, min_bs, max_bs):
#     mid = (min_bs + max_bs)/2.
#     for i in range(30):
#         if fn(mid) > 0:
#             max_bs = mid
#         else:
#             min_bs = mid
#         mid = (min_bs + max_bs)/2.
#     return mid
# 
# np.random.seed(0)
# vec = np.random.random_sample(M)
# print "delta", delta
# print "vector", vec
# proj = project(vec, delta)
# print "projection", proj, np.dot(proj, delta)
# 
# print ""
# vec = delta/(3*np.dot(vec, delta))
# print "delta", delta
# print "vector", vec, "dot-product", np.dot(vec, delta)
# proj = project(vec, delta)
# print "projection", proj, np.dot(proj, delta)




# f = generate_f(delta, vec)
# f_prime = generate_f_prime(delta, vec)
# x = np.arange(-1,1,.01)
# y = [f(xl) for xl in x]
# y_prime = [f_prime(xl) for xl in x]
# plt.plot(x,y,'r', x,y_prime,'--b',x,np.zeros(x.shape),'--k')
# plt.show()
# t_hat = bin_search(f_prime, -1, 1)
# print t_hat

dataset = DataSet()
dataset.generate_classification(n_samples = 300, n_features=10, n_informative=3, n_redundant=3, random_state = 1)
dataset.save_data('blob2.data')
dataset.generate_blob(n_samples = 300, n_features = 10, random_state= 1)
dataset.save_data('blob1.data')
