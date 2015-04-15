from mklsvmfyp.test.dataset import DataSet
from mklsvmfyp.classifier import Kernel
import numpy as np
import matplotlib.pyplot

def multiplicative_normalization(kernel_matrix):
    trace = np.trace(kernel_matrix)
    sum_all = np.sum(np.sum(kernel_matrix))
    n = np.size(kernel_matrix,0)
    return kernel_matrix/(1./n * trace - 1./(n ** 2) * sum_all)

def spherical_normalization(kernel_matrix):
    diagonals = np.diag(kernel_matrix)
    n = np.size(kernel_matrix, 0)
    for i in range(n):
        for j in range(n):
            kernel_matrix[i,j] = kernel_matrix[i,j]/(diagonals[i] * diagonals[j])
    return kernel_matrix
    
if __name__ == '__main__':
    print 'Testing using ionosphere dataset.'
    ionosphere = DataSet(dataset = 'ionosphere')
    ionosphere_data = ionosphere.retrieve_sets(random_seed=742)
    num_of_data = len(ionosphere_data[0])
    X = ionosphere_data[0]
    y = ionosphere_data[1]
    
    num_of_kernels = 15
    
    kernel_sigmas = np.zeros(num_of_kernels)
    
    bound_eigenvalues_multiplicative = np.zeros(num_of_kernels)
    bound_eigenvalues_spherical = np.zeros(num_of_kernels)
    bound_eigenvalues_original = np.zeros(num_of_kernels)
    
    largest_eigenvalues_multiplicative = np.zeros(num_of_kernels)
    largest_eigenvalues_spherical = np.zeros(num_of_kernels)
    largest_eigenvalues_original = np.zeros(num_of_kernels)
    
    for k in range(num_of_kernels):
        sigma = (3**0.5)**(k - num_of_kernels/2)
        kernel_sigmas[k] = sigma
        gaussian_kernel = Kernel.gaussian(sigma)
        gram_matrix = np.zeros((num_of_data, num_of_data))
        for i in range(num_of_data):
            for j in range(num_of_data):
                gram_matrix[i, j] = gaussian_kernel(X[i,:], X[j,:])
        
        multiplicative_gram_matrix = multiplicative_normalization(gram_matrix)
        w, v = np.linalg.eig(multiplicative_gram_matrix)
        largest_eigenvalues_multiplicative[k] = np.max(np.abs(w))
        abs_matrix = np.abs(multiplicative_gram_matrix)
        bound_eigenvalues_multiplicative[k] = np.min([np.max(np.sum(abs_matrix, 0)), np.max(np.sum(abs_matrix, 1))])
        
        spherical_gram_matrix = spherical_normalization(gram_matrix)
        w, v = np.linalg.eig(spherical_gram_matrix)
        largest_eigenvalues_spherical[k] = np.max(np.abs(w))
        abs_matrix = np.abs(spherical_gram_matrix)
        bound_eigenvalues_multiplicative[k] = np.min([np.max(np.sum(abs_matrix, 0)), np.max(np.sum(abs_matrix, 1))])
    
        w, v = np.linalg.eig(gram_matrix)
        largest_eigenvalues_original[k] = np.max(np.abs(w))
        
    print kernel_sigmas
    print largest_eigenvalues_multiplicative
    print largest_eigenvalues_spherical
    print largest_eigenvalues_original
    matplotlib.pyplot.plot(np.log10(kernel_sigmas), np.log10(largest_eigenvalues_multiplicative),'r', label='Multiplicative')
    matplotlib.pyplot.plot(np.log10(kernel_sigmas), np.log10(largest_eigenvalues_spherical), 'b', label='Spherical')
    matplotlib.pyplot.plot(np.log10(kernel_sigmas), np.log10(largest_eigenvalues_original), 'k', label='Original')
    matplotlib.pyplot.title('Largest eigenvalue of kernel matrices')
    matplotlib.pyplot.xlabel('Sigma (log10 scale)')
    matplotlib.pyplot.ylabel('Largest eigenvalue (log10 scale)')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.show()