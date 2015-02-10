'''
Created on 3 Feb, 2015

@author: ivanwangsa
'''
from mklsvmfyp.test.dataset import DataSet
from mklsvmfyp.classifier import Kernel, SilpMklSvm, SoftMargin1Svm



if __name__ == '__main__':
    print 'Testing using ionosphere dataset.'
    ionosphere = DataSet(dataset = 'ionosphere')
    (train, valid, test) = ionosphere.retrieve_sets()
    kernel_1 = Kernel.gaussian(1.)
    svm = SilpMklSvm(constraint=10., kernels=tuple([Kernel.gaussian(3. ** (i-4)) for i in range(10)]))
    svm.fit(train[0], train[1])
    print svm.score(test[0], test[1])
    
    svm2 = SoftMargin1Svm(kernel = kernel_1, constraint= 1.)
    svm2.fit(train[0], train[1])
    print svm2.score(test[0], test[1])