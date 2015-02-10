'''
Created on 28 Jan, 2015

@author: ivanwangsa
'''


from cvxopt import solvers, matrix, spmatrix
import mosek
import numpy as np
import matplotlib.pyplot as plt
import operator
import sklearn.svm
from sklearn import svm

solvers.options['show_progress'] = False
solvers.options['MOSEK'] = {mosek.iparam.log: 0}

EPS = 1e-7

class BinaryClassifier(object):
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def score(self, X, y):
        predicted_y = self.predict(X)
        return float(sum(y == predicted_y))/len(y)
    
    def visualize_2d(self, xrg = (-1.0, 1.0), yrg = (-1.0, 1.0), size = (10, 10)):
        fig, axs = plt.subplots()
        num_rows = size[0] + 1
        num_cols = size[1] + 1
        x = np.linspace(xrg[0],xrg[1],num = num_rows)
        y = np.linspace(yrg[0],yrg[1],num = num_cols)
        X, Y = np.meshgrid(x,y)
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.reshape(self.predict(positions), np.shape(X))
        
        cs = axs.contourf(X, Y, Z, levels=np.linspace(-1, 1, num = 21))
        fig.colorbar(cs, ax = axs)
        
        total_indices = np.array(range(len(self._y)))
        pos = total_indices[self._y > 0]
        neg = total_indices[self._y < 0]
        axs.plot(self._X[pos, 0], self._X[pos, 1], 'ro', mew = 1, ms = 5)
        axs.plot(self._X[neg, 0], self._X[neg, 1], 'bo', mew = 1, ms = 5)
        plt.show(block = False)
    
    def __init__(self):
        pass    

class NearestNeighbour(BinaryClassifier):
    def fit(self, X, y):
        self._X = X
        self._y = y
    
    def predict(self, X):
        num_of_obs = self._X.shape[0]
        num_of_new_obs = X.shape[0]
        y = np.zeros(num_of_new_obs)
        def norm(v1, v2):
            return np.linalg.norm(v1-v2, ord = 2)
        for i in range(num_of_new_obs):
            neighbors = []
            for j in range(num_of_obs):
                if len(neighbors) < self._num_of_neighbors or neighbors[0][0] > norm(X[i], self._X[j]):
                    if len(neighbors) < self._num_of_neighbors:
                        neighbors.append((norm(X[i], self._X[j]), j))
                    else:
                        neighbors[0] = (norm(X[i], self._X[j]), j)
                    neighbors.sort()
                    neighbors.reverse()
            for j in range(self._num_of_neighbors):
                y[i] += self._y[neighbors[j][1]]
            y[i] = np.sign(y[i])
        return np.sign(y)
    
    def __init__(self, number_of_neighbors = 3):
        super(self.__class__, self).__init__()
        self._num_of_neighbors = number_of_neighbors

class Kernel(object):
    @staticmethod
    def linear():
        def result(x, y):
            return np.dot(x.T, y)
        return result
    
    @staticmethod
    def polynomial(deg, a0 = 1):
        def result(x,y):
            return (a0 + np.dot(x.T, y)) ** deg
        return result

    @staticmethod
    def gaussian(sigma = 1.):
        def result(x,y):
            return np.exp(-np.linalg.norm(x-y)/(2 * sigma ** 2))
        return result;
    
    @staticmethod
    def sigmoid(kappa = 1., a0 = -1.):
        def result(x,y):
            return np.tanh(kappa * np.dot(x.T, y) + a0)
        return result
        
class Svm(BinaryClassifier):
    # TO DO: TOO SLOW!
    def _compute_gram_matrix(self):
        kernel = self._kernel
        X = self._X
        n = X.shape[0]
        N = np.zeros((n, n))
        for i in xrange(n):
            for j in xrange(n):
                N[i,j] = kernel(X[i], X[j])
        return N
    def __init__(self):
        pass        

class HardMarginSvm(Svm): 
    def fit(self, X, y):
        self._X = X
        self._y = y
        
        n = X.shape[0]
        yy = np.array([y]).T
        
        N = self._compute_gram_matrix()
        
        P = matrix(np.outer(y,y) * N)
        q = matrix(-1., (n,1))
        
        G = spmatrix(-1., range(n), range(n))
        h = matrix(0., (n,1))

        A = matrix(yy.T)
        b = matrix(0.)
        
        soln = solvers.qp(P,q,G,h,A,b,solver='mosek')
        
        self.dual_variables = np.hstack(np.array(soln['x']))
        
        self.support_vector_indices = np.array(xrange(n))
        self.support_vector_indices = self.support_vector_indices[self.dual_variables > EPS]
        self.support_vectors = X[self.support_vector_indices, ]
        
        pos = self.support_vector_indices[y[self.support_vector_indices] > 0]
        neg = self.support_vector_indices[y[self.support_vector_indices] < 0]
        
        self.bias = 0
        
        opt = None
        for i in neg:
            tmp = 0
            for j in self.support_vector_indices:
                tmp += self.dual_variables[j] * y[j] * np.dot(X[i,], X[j,].T)
            if opt == None or opt < tmp:
                opt = tmp        
        self.bias += opt        
        opt = None        
        for i in pos:
            tmp = 0
            for j in self.support_vector_indices:
                tmp += self.dual_variables[j] * y[j] * np.dot(X[i,], X[j,].T)
            if opt == None or opt > tmp:
                opt = tmp        
        self.bias += opt     
        
        self.bias /= -2
    
    def predict(self, X):
        num_of_new_obs = X.shape[0]
        signals = np.zeros(num_of_new_obs)
        for i in range(num_of_new_obs):
            signals[i] = self.bias
            for j in self.support_vector_indices:
                signals[i] += self.dual_variables[j] * self._y[j] * self._kernel(X[i], self._X[j])
        return np.sign(signals)
    
    def __init__(self, kernel = Kernel.gaussian(1.0)):
        super(self.__class__, self).__init__()
        self._kernel = kernel
        

class SoftMargin1Svm(Svm):
    def fit(self, X, _y):
        self._X = X
        self._y = _y
        
        n = X.shape[0]
        yy = np.array([_y]).T
        
        N = self._compute_gram_matrix()
        
        P = matrix(np.outer(_y,_y) * N)
        q = matrix(-1., (n,1))
        
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = matrix(np.repeat((0., self._constraint), n), (2*n, 1))

        A = matrix(yy.T)
        b = matrix(0.)
        
        soln = solvers.qp(P,q,G,h,A,b,solver='mosek')
        
        self.dual_variables = np.hstack(np.array(soln['x']))
        
        self.support_vector_indices = np.array(xrange(n))
        self.support_vector_indices = self.support_vector_indices[self.dual_variables > EPS]
        self.support_vectors = X[self.support_vector_indices, ]
        
        self.bias = 0
        
        nonbound_points_numbers = 0
        
        for k in self.support_vector_indices:
            if self.dual_variables[k] < self._constraint - EPS:
                nonbound_points_numbers += 1
                this_bias = self._y[k]
                for i in range(n):
                    this_bias -= self.dual_variables[i] * self._y[i] * self._kernel(X[i], X[k])
                self.bias += this_bias
        try:
            self.bias /= nonbound_points_numbers
        except ZeroDivisionError:
            print 'Zero nonbound points found. Consider increasing the constraint penalty.'

    def predict(self, X):
        num_of_new_obs = X.shape[0]
        signals = np.zeros(num_of_new_obs)
        for i in range(num_of_new_obs):
            signals[i] = self.bias
            for j in self.support_vector_indices:
                signals[i] += self.dual_variables[j] * self._y[j] * self._kernel(X[i], self._X[j])
        return np.sign(signals)
    
    def __init__(self, kernel = Kernel.gaussian(1.0), constraint = 1.0):
        super(self.__class__, self).__init__()
        self._constraint = constraint
        self._kernel = kernel
        
    
class SilpMklSvm(Svm):
    # uses scikit-learn
    
    def _compute_gram_matrices(self):
        X = self._X
        n = self._X.shape[0]
        kernels = self._kernels
        res = [np.matrix(np.zeros((n,n))) for i in range(len(kernels))]
        for m in range(len(kernels)):
            for i in xrange(n):
                for j in xrange(n):
                    res[m][i,j] = kernels[m](X[i], X[j])
        self._gram_matrices = tuple(res)
    
    def fit(self, X, y):
        # TODO: Finish this!
        self._X = X
        self._y = y
        self._compute_gram_matrices()
        K = len(self._gram_matrices)
        N = X.shape[0]
        SILP_EPS = 1e-6
        
        def S_k(support_, dual_coef_, k):
            # support_ are of the indices of support vectors
            # dual_coef_ are alpha * y
            
            combined = zip(support_, dual_coef_)
            combined.sort()
            combined = zip(*combined) # unzip combined
            # combined[0] : support indices, combined[1] : dual_coef_
            indices = combined[0]
            kernel_matrix = self._gram_matrices[k]
            subset_kernel_matrix = kernel_matrix[indices, :][:, indices]
            dual_coef = np.matrix(combined[1]).T
            #print subset_kernel_matrix
            #print '--------------'
            #print np.matrix(combined[1]), matrix[combined[0],:][:,combined[0]]
            #print '--------------'
            res = 1.0/2 * dual_coef.T * subset_kernel_matrix * dual_coef - np.sum(np.absolute(dual_coef))
            return res[0,0]
        
        num_of_rows_of_S = 2
        S = np.zeros((num_of_rows_of_S, K))
        theta = -1e9
        beta = np.repeat(1./K, K)
        t = 0
        self.single_svm_solver = svm.SVC(C = self._constraint, kernel = 'precomputed')
        
        # prepare matrices for LP
        A_1 = np.hstack(([[0.]], np.ones((1, K))))
        A_1 = np.vstack((A_1, -A_1))
        A_2 = np.hstack( (np.zeros( (K, 1) ), -np.eye(K) ) )
        A = np.vstack((A_1, A_2))
        c = np.vstack(([[-1.]], np.zeros((K, 1))))
               
        while True:
            gram_matrix = np.matrix(np.zeros((N, N)))
            for k in range(K):
                gram_matrix += beta[k] * self._gram_matrices[k]
            self.single_svm_solver.fit(gram_matrix, self._y)
            dual_coef = self.single_svm_solver.dual_coef_[0]
            supports = self.single_svm_solver.support_
            if t == num_of_rows_of_S:
                S = np.vstack((S, np.zeros(S.shape)))
                num_of_rows_of_S *= 2
            S[t,] = np.array([S_k(supports, dual_coef, i) for i in range(K)])
            S_now = np.dot(beta, S[t,])
            if np.abs(1-S_now/theta) < SILP_EPS:
                break
            additional_A = np.hstack( ([1.], -S[t,:] ) )
            t += 1
            A = np.vstack((A, additional_A))
            b = np.vstack((np.array([[1.], [-1.]]), np.zeros((K+t, 1))))
            sol = solvers.lp(matrix(c), matrix(A), matrix(b), solver='mosek')
            soln = sol['x']
            theta = soln[0]
            beta = np.ravel(soln[1:K+1])           
        
        gram_matrix = np.matrix(np.zeros((N, N)))
        for k in range(K):
            gram_matrix += beta[k] * self._gram_matrices[k]
        
        self.single_svm_solver.fit(gram_matrix, self._y)
        self._beta = beta
                
    def predict(self, X):
        num_row_test = X.shape[0]
        num_row_train = self._X.shape[0]
        kernel_matrix = np.zeros((num_row_test, num_row_train))
        for i in xrange(num_row_test):
            for j in xrange(num_row_train):
                kernel_matrix[i,j] = sum([self._beta[k] * self._kernels[k](X[i, ], self._X[j,]) for k in range(len(self._kernels))])
        return np.array(self.single_svm_solver.predict(kernel_matrix))
            
    def __init__(self, kernels, constraint = 1., epsilon = 0.01):
        self._kernels = kernels
        self._constraint = constraint
        self._epsilon = epsilon
    
        