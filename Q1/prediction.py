"""
Functions for predicting the output
"""
import time
import numpy as np
from preprocessing import json_reader
from sklearn.metrics import f1_score

def predictClass (Phi, Theta, dictionary, doc):
    """
    Actual Predictions
    """

    probs = np.zeros(5)
    for k in range(0, 5):
        probs[k] += Phi[k]
        for word in doc:
            if word not in dictionary:
                continue
            w = dictionary[word]
            probs[k] += Theta[w][k]
    return np.argmax(probs) + 1


def actualPredictions (Phi, Theta, testset, dictionary, count=1000, accuracyLabel="Prediction Accuracy: "):
    """
    Predictions from training
    """

    tick = time.time()

    correctPredictions = 0
    totalPredictions = 0
    for data in json_reader(testset, count):
        prediction = predictClass (Phi, Theta, dictionary, data['review'])
        totalPredictions += 1
        # print (prediction, end=' ')
        if (prediction == data['rating']):
            correctPredictions += 1
    # print()
    print (totalPredictions)
    print (accuracyLabel, "%.2f%%" % (correctPredictions * 100 / totalPredictions))
    print ("Time Taken: ", time.time() - tick)


def randomPredictions (testset, count=1000):
    """
    Random Predictions
    """
    tick = time.time()

    correctPredictions = 0
    totalPredictions = 0
    for data in json_reader(testset, count):
        prediction = np.random.randint(1, 6) 
        totalPredictions += 1
        if (prediction == data['rating']):
            correctPredictions += 1
        
    print (correctPredictions, totalPredictions, count)
    print ("Random Prediction Accuracy: %.2f%%" % (correctPredictions * 100 / totalPredictions))
    print ("Time Taken: ", time.time() - tick)


def majorityPrediction (testset, count=1000):
    """
    Majority Prediction
    """
    tick = time.time()

    correctPredictions = 0
    totalPredictions = 0
    for data in json_reader(testset, count):
        prediction = 5
        totalPredictions += 1
        if (prediction == data['rating']):
            correctPredictions += 1
        
    print (correctPredictions, totalPredictions, count)
    print ("Majority Prediction Accuracy: %.2f%%" % (correctPredictions * 100 / totalPredictions))
    print ("Time Taken: ", time.time() - tick)


def predictionArray (Phi, Theta, testset, dictionary, count=1000):
    """
    Predictions from training
    """

    tick = time.time()

    ratings = []
    predictions = []
    for data in json_reader(testset, count):
        prediction = predictClass (Phi, Theta, dictionary, data['review'])
        ratings.append(data['rating'] - 1)
        predictions.append(prediction - 1)
        
    print ("Time Taken: ", time.time() - tick)
    return (ratings, predictions)


def F1scores (Phi, Theta, testset, dictionary, count=1000):
    tick = time.time()

    (ratings, predictions) = predictionArray (Phi, Theta, testset, dictionary, count)
    F1_scores = f1_score(ratings, predictions, average=None)
    macroF1 = f1_score(ratings, predictions, average='macro')

    print ("Class   F1 Score")
    for c in range(0, 5):
        print ("  %d       %.3f" % (c+1, F1_scores[c]))
    print ("Macro F1 Score: %.3f" % (macroF1))
    print ("Time taken: ", time.time() - tick)