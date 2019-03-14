"""
Code for training with the LIBSVM package
"""

import time
import numpy as np
import pickle
from libsvm.python.svmutil import *
from plot import plotConfusionMatrix

def train (X, Y, kernel="linear", gamma=0.05, C=1):
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
    param.C = C
    # param.H = 0
    model = svm_train(prob, param)

    print ("Time Taken (LIBSVM %s kernel): " % (kernel), time.time() - tick)
    return model


def predict (X, Y, model):
    return svm_predict(Y, X, model)[0]


def findAccuracy (predictions, Y):
    return np.sum(predictions == Y) / len(Y)


def binary (X, Y, testX, testY, kernel="linear", gamma=0.05):
    model = train (X, Y, kernel, gamma)

    # Training Accuracy
    predictions = predict (X, Y, model)
    trainAccuracy = findAccuracy (predictions, Y)
    print ("Training Accuracy: ", trainAccuracy)

    # Testing Accuracy
    predictions = predict (testX, testY, model)
    testAccuracy = findAccuracy (predictions, testY)
    print ("Test Accuracy: ", testAccuracy)

    return


def multi (Data, testX, testY):
    """
    Train nC2 models and do predictions
    """
    tick = time.time()

    testX = np.array(testX)
    testY = np.array(testY)

    Models = []
    for i in range (10):
        Models.append([])
        for j in range (10):
            Models[i].append(0)
        for j in range (i+1, 10):
            (X, Y) = Data[i][j]
            Models[i][j] = train(X, Y, kernel="gaussian")
    
    predictions = []
    for i in range (10):
        for j in range (i+1, 10):
            def binaryFi (y):
                if (y == 1):
                    return i
                else:
                    return j
            vf = np.vectorize(binaryFi)
            pred = predict (testX, testY, Models[i][j])
            predictions.append(vf(pred))

    predictions = np.array(predictions)
    print (predictions)

    axis = 0
    u, indices = np.unique(predictions, return_inverse=True)
    finalPredictions = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(predictions.shape),
                                None, np.max(indices) + 1), axis=axis)]

    print (finalPredictions, u)
    predictionsFile = 'Q2/models/predictions45.pred'
    with open(predictionsFile, 'wb') as handle:
        pickle.dump(finalPredictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    accuracy = np.sum(finalPredictions == testY) / len (testY)
    print (accuracy, np.sum(finalPredictions == testY), len (testY))
    print ("Time taken: ", time.time() - tick)


def confusionMatrix (testY):
    predictionsFile = 'Q2/models/predictions45.pred'
    with open(predictionsFile, 'rb') as handle:
        predictions = pickle.load(handle)

    plotConfusionMatrix (predictions, testY)


def validationC (Data, valX, valY, testX, testY):
    """
    Train nC2 models for different C's and do predictions
    """

    C = [1e-5, 1e-3, 1, 5, 10]
    Accuracies = []
    print (C)

    tick = time.time()

    testX = np.array(testX)
    testY = np.array(testY)
    valX = np.array(valX)
    valY = np.array(valY)

    for c in C:
        Models = []
        for i in range (10):
            Models.append([])
            for j in range (10):
                Models[i].append(0)
            for j in range (i+1, 10):
                (X, Y) = Data[i][j]
                Models[i][j] = train(X, Y, kernel="gaussian", C=c)

        def predictTest (testX, testY):
            predictions = []
            for i in range (10):
                for j in range (i+1, 10):
                    def binaryFi (y):
                        if (y == 1):
                            return i
                        else:
                            return j
                    vf = np.vectorize(binaryFi)
                    pred = predict (testX, testY, Models[i][j])
                    predictions.append(vf(pred))

            predictions = np.array(predictions)
            print (predictions)

            axis = 0
            u, indices = np.unique(predictions, return_inverse=True)
            finalPredictions = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(predictions.shape),
                                        None, np.max(indices) + 1), axis=axis)]

            print (finalPredictions, u)
            accuracy = np.sum(finalPredictions == testY) / len (testY)
            return accuracy

        valAccuracy = predictTest(valX, valY)
        testAccuracy = predictTest (testX, testY)
        Accuracies.append((valAccuracy, testAccuracy))

    for idx, c in enumerate(C):
        print ("C = %f | Val. Accuracy = %f | Test Accuracy = %f" % (c, Accuracies[idx][0], Accuracies[idx][1] ))
    
    print ("Time taken: ", time.time() - tick)
