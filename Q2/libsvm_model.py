"""
Code for training with the LIBSVM package
"""

import time
import numpy as np
from libsvm.python.svmutil import *

def binary (X, Y, testX, testY, kernel="linear", gamma=0.05):
    """
    Train SVM with linear kernel
    """

    tick = time.time()

    prob = svm_problem (Y, X)
    param = svm_parameter()
    if kernel == "linear":
        param.kernel_type = LINEAR
    else:
        param.kernel_type = RBF
        param.gamma = gamma
    param.C = 1
    model = svm_train(prob, param)

    print ("Time Taken (LIBSVM %s kernel): " % (kernel), time.time() - tick)
    predictions = svm_predict(Y, X, model)[0]
    print ("Training Accuracy: ", np.sum(predictions == Y) / len(Y))
    predictions = svm_predict(testY, testX, model)[0]
    print ("Test Accuracy: ", np.sum(predictions == testY) / len(testY))

    return np.sum(predictions == testY) / len(testY)

