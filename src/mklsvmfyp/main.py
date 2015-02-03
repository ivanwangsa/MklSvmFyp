'''
Created on 3 Feb, 2015

@author: ivanwangsa
'''
from mklsvmfyp.test.dataset import DataSet
from mklsvmfyp.classifier import Kernel, SilpMklSvm



if __name__ == '__main__':
    print 'Testing using ionosphere dataset.'
    ionosphere = DataSet(dataset = 'ionosphere')
    (train, valid, test) = ionosphere.retrieve_sets()
    
    kernel_1 = Kernel.gaussian(1.)
    kernel_2 = Kernel.gaussian(0.1)
    
    svm = SilpMklSvm(constraint=1., kernels=(kernel_1, kernel_2))
    svm.fit(train[0], train[1])
