'''
Created on 3 Feb, 2015

@author: ivanwangsa
'''
from mklsvmfyp.test.dataset import DataSet
from mklsvmfyp.classifier import Kernel, SilpMklSvm, SoftMargin1Svm,\
    ModifiedSimpleMklSvm, SimpleMklSvm
from sklearn import svm



if __name__ == '__main__':
    print 'Testing using ionosphere dataset.'
    ionosphere = DataSet(dataset = 'ionosphere')
    ionosphere_data = ionosphere.retrieve_sets(random_seed=756765)
    num_of_data = len(ionosphere_data[0])
    train = (ionosphere_data[0][0:2*num_of_data/3], ionosphere_data[1][0:2*num_of_data/3])
    test = (ionosphere_data[0][2*num_of_data/3:], ionosphere_data[1][2*num_of_data/3:])
    kernel_1 = Kernel.gaussian(1.)
    kernel_2 = Kernel.gaussian(.5)
    kernel_3 = Kernel.gaussian(2.)
    kernel_4 = Kernel.gaussian(.1)
    kernel_5 = Kernel.linear()
    
    silpmklsvm = SilpMklSvm(constraint=7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5))
    silpmklsvm.fit(train[0], train[1])
    print 'SILP-MKLSVM'
    print silpmklsvm.score(train[0], train[1])
    print silpmklsvm.score(test[0], test[1])
#     
    simpleMklSvm = SimpleMklSvm(constraint=7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5))
    simpleMklSvm.fit(train[0], train[1])
    print 'SimpleMKLSVM'
    print simpleMklSvm.score(train[0], train[1])
    print simpleMklSvm.score(test[0], test[1])
    
    simplesvm = svm.SVC(C = 1., kernel='linear')
    simplesvm.fit(train[0], train[1])
    print 'Standard SVM'
    print simplesvm.score(train[0], train[1])
    print simplesvm.score(test[0], test[1])
#     
    projectedSimpleMkl = ModifiedSimpleMklSvm(constraint = 7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5), method = 'projected')
    projectedSimpleMkl.fit(train[0], train[1])
    print 'Projected MKLSVM'
    print projectedSimpleMkl.score(train[0], train[1])
    print projectedSimpleMkl.score(test[0], test[1])
#     
    reducedSimpleMkl = ModifiedSimpleMklSvm(constraint = 7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5))
    reducedSimpleMkl.fit(train[0], train[1])
    print 'Reduced MKLSVM'
    print reducedSimpleMkl.score(train[0], train[1])
    print reducedSimpleMkl.score(test[0], test[1])
    
    
    