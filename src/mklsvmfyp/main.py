'''
Created on 3 Feb, 2015

@author: ivanwangsa
'''
from mklsvmfyp.test.dataset import DataSet
from mklsvmfyp.classifier import Kernel, SilpMklSvm, SoftMargin1Svm
from sklearn import svm



if __name__ == '__main__':
    print 'Testing using ionosphere dataset.'
    ionosphere = DataSet(dataset = 'ionosphere')
    ionosphere_data = ionosphere.retrieve_sets(random_seed=1023)
    num_of_data = len(ionosphere_data[0])
    train = (ionosphere_data[0][0:2*num_of_data/3], ionosphere_data[1][0:2*num_of_data/3])
    test = (ionosphere_data[0][2*num_of_data/3:], ionosphere_data[1][2*num_of_data/3:])
    kernel_1 = Kernel.gaussian(1.)
    kernel_2 = Kernel.gaussian(.5)
    kernel_3 = Kernel.gaussian(2.)
    kernel_4 = Kernel.gaussian(.1)
    kernel_5 = Kernel.linear()
    mklsvm = SilpMklSvm(constraint=7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5))
    mklsvm.fit(train[0], train[1])
    print mklsvm.score(train[0], train[1])
    print mklsvm.score(test[0], test[1])
    
    svm2 = SoftMargin1Svm(kernel = kernel_5, constraint= 1.)
    svm2.fit(train[0], train[1])
    print svm2.score(train[0], train[1])
    print svm2.score(test[0], test[1])
    
    simplesvm = svm.SVC(C = 1., kernel='linear')
    simplesvm.fit(train[0], train[1])
    print simplesvm.score(train[0], train[1])
    print simplesvm.score(test[0], test[1])