"""
Train the NB classifier in multiple ways
"""

import sys
import json
import time
import pickle
import numpy as np
from preprocessing import json_reader, loadVocab
from prediction import actualPredictions, randomPredictions, majorityPrediction, F1scores
from plot import plotConfusionMatrix

# def computePHI (trainset, count=1000):
#     """
#     Computing PHI_k = P (y = k)
#     """
#     Phi = np.zeros(5)
#     tick = time.time()

#     totalRatings = 0
#     for data in json_reader(trainset, count):
#         totalRatings += 1
#         Phi[data['rating'] - 1] += 1
        
#     Phi = np.log(Phi / totalRatings)
#     print ("PHI -- Time Taken: ", time.time() - tick)
#     return Phi


def computeParameters (trainset, dictionary, count=1000, stemming=False, bigrams=False):
    # Computing ThetaWK's
    V = len(dictionary)
    print (V)

    tick = time.time()

    Phi = np.zeros(5)
    ThetaNum = np.zeros((V, 5)) + 1
    ThetaDeno = np.zeros(5) + V
    totalRatings = [0]
    
    def computeFreq (doc, rating):
        totalRatings[0] += 1

        k = rating - 1
        Phi[k] += 1

        Deno = len(doc)
        
        for word in doc:
            if (word not in dictionary):
                Deno -= 1
                continue
            w = dictionary[word]
            ThetaNum[w][k] += 1

        ThetaDeno[k] += Deno
        
        return 0
        
    for data in json_reader(trainset, count, stemming=stemming, bigrams=bigrams):
        computeFreq (data['review'], data['rating'])
        
    Phi = np.log(Phi / totalRatings[0])
    Theta = np.log(ThetaNum / ThetaDeno)

    np.set_printoptions(precision=2)

    print ("Parameters -- Time Taken: ", time.time() - tick)
    return Phi, Theta


def saveAndLoadModel (mode, PHI=None, THETA=None, modelName='Q1/models/NBmodel.pickle'):
    """
    0 for saving the model
    1 for loading the model
    """

    if (mode == 0):
        model = (PHI, THETA)
        with open(modelName, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(modelName, 'rb') as handle:
            (PHI, THETA) = pickle.load(handle)
        return (PHI, THETA)


if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print ("Go Away")
        exit(1)

    trainset = sys.argv[1]
    testset = sys.argv[2]
    part = sys.argv[3]
    count = 2000 # Upper limit on number of examples to consider

    if (part == 'a'):
        # Implement NB on smaller data
        dictionary = loadVocab('Q1/vocabs/unigramVocab.pickle')
        # PHI = computePHI(trainset, count)
        PHI, THETA = computeParameters (trainset, dictionary, count, False)
        actualPredictions(PHI, THETA, trainset, dictionary, count, accuracyLabel="Training Accuracy: ")
        actualPredictions(PHI, THETA, testset, dictionary, count, accuracyLabel="Test Accuracy: ")
        saveAndLoadModel (0, PHI, THETA, modelName='Q1/models/NBsimple.pickle')
        exit (0)

    if (part == 'b'):
        # Random guessing and majority prediction
        randomPredictions (testset, count)
        majorityPrediction (testset, count)
        exit(0)

    if (part == 'c'):
        # Plot the confusion matrix
        dictionary = loadVocab('Q1/vocabs/unigramVocab.pickle')
        (PHI, THETA) = saveAndLoadModel (1, modelName='Q1/models/NBsimple.pickle')
        plotConfusionMatrix (PHI, THETA, testset, dictionary, count)
        exit(0)

    if (part == 'd'):
        # Implement NB on smaller data
        dictionary = loadVocab('stemmedVocab.pickle')
        # PHI = computePHI(trainset, count)
        PHI, THETA = computeParameters (trainset, dictionary, count, True)
        actualPredictions(PHI, THETA, trainset, dictionary, count, accuracyLabel="Training Accuracy: ")
        actualPredictions(PHI, THETA, testset, dictionary, count, accuracyLabel="Test Accuracy: ")
        saveAndLoadModel (0, PHI, THETA, modelName='Q1/models/NBstemmed.pickle')
        exit (0)

    if (part == 'e'):
        # Implement NB on smaller data
        dictionary = loadVocab('Q1/vocabs/bigramVocab.pickle')
        # PHI = computePHI(trainset, count)
        PHI, THETA = computeParameters (trainset, dictionary, count, False, bigrams=True)
        actualPredictions(PHI, THETA, trainset, dictionary, count, accuracyLabel="Training Accuracy: ")
        actualPredictions(PHI, THETA, testset, dictionary, count, accuracyLabel="Test Accuracy: ")
        saveAndLoadModel (0, PHI, THETA, modelName='Q1/models/NBfeature.pickle')
        exit (0)

    if (part == 'f'):
        # Report F1-scores
        dictionary = loadVocab('Q1/vocabs/unigramVocab.pickle')
        (PHI, THETA) = saveAndLoadModel (1)
        F1scores (PHI, THETA, testset, dictionary, count)
        exit(0)

    print("Go Away")