'''
Created on 28 Jan, 2015

@author: ivanwangsa
'''


from cvxopt import solvers, matrix, spmatrix
import mosek
import numpy as np
import matplotlib.pyplot as plt
import operator

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
        plt.show()
    
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
    def gaussian(sigma):
        def result(x,y):
            return np.exp(-np.linalg.norm(x-y)/(2 * sigma ** 2))
        return result;
    
    @staticmethod
    def sigmoid(kappa, a0 = 0):
        def result(x,y):
            return np.tanh(kappa * np.dot(x.T, y) + a0)
        return result
        
class Svm(BinaryClassifier):
    def _compute_gram_matrix(self):
        n = self._X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self._kernel(self._X[i], self._X[j])
        return K

class HardMarginSvm(Svm): 
    def fit(self, X, y):
        self._X = X
        self._y = y
        
        n = X.shape[0]
        yy = np.array([y]).T
        
        K = self._compute_gram_matrix()
        
        P = matrix(np.outer(y,y) * K)
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
                tmp += self.dual_variables[j]*y[j]*np.dot(X[i,], X[j,].T)
            if opt == None or opt < tmp:
                opt = tmp        
        self.bias += opt        
        opt = None        
        for i in pos:
            tmp = 0
            for j in self.support_vector_indices:
                tmp += self.dual_variables[j]*y[j]*np.dot(X[i,], X[j,].T)
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
        self._kernel = kernel
        

class SoftMargin1Svm(Svm):
    def fit(self, X, _y):
        self._X = X
        self._y = _y
        
        n = X.shape[0]
        yy = np.array([_y]).T
        
        K = self._compute_gram_matrix()
        
        P = matrix(np.outer(_y,_y) * K)
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
        
        for k in range(n):
            if self.dual_variables[k] > EPS and self.dual_variables[k] < self._constraint - EPS:
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
        self._constraint = constraint
        self._kernel = kernel
        
class SoftMargin2Svm(Svm):
    def fit(self, X, y):
        self._hard_margin_svm.fit(X, y)
    
    def predict(self, X):
        return self._hard_margin_svm.predict(X)
        
    def __init__(self, kernel = Kernel.gaussian(1.0), constraint = 1.0):
        def new_kernel(x, y):
            if reduce(operator.and_, np.equal(x, y)):
                return 1/(2*constraint) + kernel(x,y)
            else:
                return kernel(x,y)
        self._hard_margin_svm = HardMarginSvm(new_kernel)
        
class SilpMklSoftMargin1Svm(Svm):
    def __init__(self, kernels, constraint):
        self._kernels = kernels
        self._constraint = constraint
        