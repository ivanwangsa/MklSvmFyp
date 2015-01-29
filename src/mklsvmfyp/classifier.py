'''
Created on 28 Jan, 2015

@author: ivanwangsa
'''


from cvxopt import solvers, matrix, spmatrix
import mosek
import numpy as np
import matplotlib.pyplot as plt

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
    
    def visualize(self, xrg = (-1.0, 1.0), yrg = (-1.0, 1.0), size = (10, 10)):
        num_rows = size[0] + 1
        num_cols = size[1] + 1
        x = np.linspace(xrg[0],xrg[1],num = num_rows)
        y = np.linspace(yrg[0],yrg[1],num = num_cols)
        X, Y = np.meshgrid(x,y)
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.reshape(self.predict(positions), np.shape(X))
        
        plt.figure()
        plt.contour(X, Y, Z, levels=[ -EPS, EPS])
        plt.show()
    
    def __init__(self):
        pass
    

class NearestNeighbour(BinaryClassifier):
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        number_of_observations = self.X.shape[0]
        number_of_new_observations = X.shape[0]
        y = np.zeros(number_of_new_observations)
        def norm(v1, v2):
            return np.linalg.norm(v1-v2, ord = 2)
        for i in range(number_of_new_observations):
            neighbors = []
            for j in range(number_of_observations):
                if len(neighbors) < self._number_of_neighbors or neighbors[0][0] > norm(X[i], self.X[j]):
                    if len(neighbors) < self._number_of_neighbors:
                        neighbors.append((norm(X[i], self.X[j]), j))
                    else:
                        neighbors[0] = (norm(X[i], self.X[j]), j)
                    neighbors.sort()
                    neighbors.reverse()
            for j in range(self._number_of_neighbors):
                y[i] += self.y[neighbors[j][1]]
            y[i] = np.sign(y[i])
        return np.sign(y)
    
    def __init__(self, number_of_neighbors = 3):
        self._number_of_neighbors = number_of_neighbors
        
class Svm(BinaryClassifier):
    pass


