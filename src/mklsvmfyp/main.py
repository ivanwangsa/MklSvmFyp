'''
Created on 3 Feb, 2015

@author: ivanwangsa
'''
from dataset import DataSet
from classifier import Kernel, SilpMklSvm, SoftMargin1Svm,\
     SimpleMklSvm, PriorMklSvm
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
    kernel_6 = Kernel.polynomial(2, 1)
#     kernel_7 = Kernel.polynomial(3, 1)
    
    list_kernels = [kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6]
    list_kernels = tuple(list_kernels)
#     silpmklsvm = SilpMklSvm(constraint=7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5))
#     silpmklsvm.fit(train[0], train[1])
#     print 'SILP-MKLSVM'
#     print silpmklsvm.score(train[0], train[1])
#     print silpmklsvm.score(test[0], test[1])
#     
#     simpleMklSvm = SimpleMklSvm(constraint=7., kernels=list_kernels)
#     simpleMklSvm.fit(train[0], train[1])
#     print 'SimpleMKLSVM'
#     print simpleMklSvm.score(train[0], train[1])
#     print simpleMklSvm.score(test[0], test[1])
#     print simpleMklSvm.kernel_coefficients
#     simplesvm = svm.SVC(C = 1., kernel='linear')
#     simplesvm.fit(train[0], train[1])
#     print 'Standard SVM'
#     print simplesvm.score(train[0], train[1])
#     print simplesvm.score(test[0], test[1])
#     
    priorMkl = PriorMklSvm(constraint = 7., kernels=list_kernels, normalize_kernels=False)
#     priorMkl.fit(train[0], train[1])
#     print 'Prior MKLSVM'
#     print priorMkl.score(train[0], train[1])
#     print priorMkl.score(test[0], test[1])
#     print priorMkl.kernel_coefficients
    
    priorMkl.set_method('conditional')
    priorMkl.fit(train[0], train[1])
    print 'Prior MKLSVM'
    print priorMkl.score(train[0], train[1])
    print priorMkl.score(test[0], test[1])
    print priorMkl.kernel_coefficients
#     
#     reducedSimpleMkl = ModifiedSimpleMklSvm(constraint = 7., kernels=(kernel_1, kernel_2, kernel_3, kernel_4, kernel_5))
#     reducedSimpleMkl.fit(train[0], train[1])
#     print 'Reduced MKLSVM'
#     print reducedSimpleMkl.score(train[0], train[1])
#     print reducedSimpleMkl.score(test[0], test[1])
    
    
    