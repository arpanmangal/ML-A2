"""
Train the NB classifier in multiple ways
"""

import sys
import json
import time
import pickle
import numpy as np
from preprocessing import json_reader, loadVocab
from prediction import actualPredictions, randomPredictions, majorityPrediction
from plot import plotConfusionMatrix

def computePHI (trainset, count=1000):
    """
    Computing PHI_k = P (y = k)
    """
    Phi = np.zeros(5)
    tick = time.time()

    totalRatings = 0
    for data in json_reader(trainset, count):
        totalRatings += 1
        Phi[data['rating'] - 1] += 1
        
    Phi = np.log(Phi / totalRatings)
    print ("PHI -- Time Taken: ", time.time() - tick)
    return Phi


def computeTHETA (trainset, dictionary, count=1000, stemming=False):
    # Computing ThetaWK's
    V = len(dictionary)
    print (V)

    tick = time.time()

    ThetaNum = np.zeros((V, 5)) + 1
    ThetaDeno = np.zeros((V, 5)) + V 

    def computeFreq (doc, rating):
        k = rating - 1
        ThetaDeno[:, k] += len(doc)
        
        for word in doc:
            if (word not in dictionary):
                ThetaDeno[:, k] -= 1
                continue
            w = dictionary[word]
            ThetaNum[w][k] += 1
        
        return 0
        
    for data in json_reader(trainset, count, stemming=stemming):
        computeFreq (data['review'], data['rating'])
        
    Theta = np.log(ThetaNum / ThetaDeno)

    np.set_printoptions(precision=2)
    # print (ThetaNum)
    # print (ThetaDeno)
    # print (Theta)

    print ("THETA -- Time Taken: ", time.time() - tick)
    return Theta


def saveAndLoadModel (mode, PHI=None, THETA=None):
    """
    0 for saving the model
    1 for loading the model
    """

    if (mode == 0):
        model = (PHI, THETA)
        with open('NBmodel.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('NBmodel.pickle', 'rb') as handle:
            (PHI, THETA) = pickle.load(handle)
        return (PHI, THETA)


if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print ("Go Away")
        exit(1)

    trainset = sys.argv[1]
    testset = sys.argv[2]
    part = sys.argv[3]
    count = 1000 # Upper limit on number of examples to consider

    if (part == 'a'):
        # Implement NB on smaller data
        dictionary = loadVocab('unigramVocab.pickle')
        PHI = computePHI(trainset, count)
        THETA = computeTHETA (trainset, dictionary, count, False)
        actualPredictions(PHI, THETA, trainset, dictionary, count, accuracyLabel="Training Accuracy: ")
        actualPredictions(PHI, THETA, testset, dictionary, count, accuracyLabel="Test Accuracy: ")
        saveAndLoadModel (0, PHI, THETA)
        exit (0)

    if (part == 'b'):
        # Random guessing and majority prediction
        randomPredictions (testset, count)
        majorityPrediction (testset, count)
        exit(0)

    if (part == 'c'):
        # Plot the confusion matrix
        dictionary = loadVocab('unigramVocab.pickle')
        (PHI, THETA) = saveAndLoadModel (1)
        plotConfusionMatrix (PHI, THETA, testset, dictionary, count)
        exit(0)