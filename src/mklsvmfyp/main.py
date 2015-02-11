'''
Created on 3 Feb, 2015

@author: ivanwangsa
'''
from mklsvmfyp.test.dataset import DataSet
from mklsvmfyp.classifier import Kernel, SilpMklSvm, SoftMargin1Svm



if __name__ == '__main__':
    print 'Testing using ionosphere dataset.'
    ionosphere = DataSet(dataset = 'ionosphere')
    ionosphere_data = ionosphere.retrieve_sets()
    num_of_data = len(ionosphere_data[0])
    train = (ionosphere_data[0][0:2*num_of_data/3], ionosphere_data[1][0:2*num_of_data/3])
    test = (ionosphere_data[0][2*num_of_data/3:], ionosphere_data[1][2*num_of_data/3:])
    kernel_1 = Kernel.gaussian(1.)
    kernel_2 = Kernel.gaussian(3.)
    kernel_3 = Kernel.gaussian(.3)
    kernel_5 = Kernel.linear()
    svm = SilpMklSvm(constraint=4., kernels=(kernel_1, kernel_2, kernel_3, kernel_5))
    svm.fit(train[0], train[1])
    print svm.score(train[0], train[1])
    print svm.score(test[0], test[1])
    
    svm2 = SoftMargin1Svm(kernel = kernel_1, constraint= 1.)
    svm2.fit(train[0], train[1])
    print svm2.score(train[0], train[1])
    print svm2.score(test[0], test[1])