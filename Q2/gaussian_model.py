"""
Code for SVM with gaussian kernel
"""

import time
import numpy as np
import pickle
import cvxopt as cvx

def binary (X, Y, testX, testY):
    """
    Train SVM with gaussian kernel
    """
    train (X, Y)
    
    predictions = predict (X, Y, X, Y)
    accuracy = findAccuracy (predictions, Y)
    print ("Training Accuracy: ", accuracy)

    predictions = predict (X, Y, testX, testY)
    accuracy = findAccuracy (predictions, testY)
    print ("Test Accuracy: ", accuracy)


    
def train (X, Y, gamma=0.05):
    """
    Train SVM with gaussian kernel
    """
    X = np.matrix(X)
    Y = np.matrix(Y).T

    m, n = X.shape

    # Find the Kernel Matrix KM
    KM = gaussianKM (X, X, gamma)

    # Parameters for CVXOPT
    YQ = Y * Y.T
    Q = np.multiply (YQ, KM)
    p = np.matrix(-np.ones((m, 1)))
    G = np.matrix(np.vstack( (-np.identity(m), np.identity(m)) ))
    h = np.matrix(np.vstack( (np.zeros((m,1)), np.ones((m,1))) ))
    A = Y.T
    b = 0
    
    # Running CVXOPT
    Q = cvx.matrix(Q)
    p = cvx.matrix(p)
    G = cvx.matrix(G)
    h = cvx.matrix(h)
    A = cvx.matrix(A, (1, m), 'd')
    b = cvx.matrix(b, (1,1), 'd')
    sol = cvx.solvers.qp(P=Q, q=p, G=G, h=h, A=A, b=b)

    # Alphas
    alphas = np.matrix(sol['x'])

    # Finding the bias
    def findBias ():
        epsilon = 1e-5
        for idx, alp in enumerate(alphas):
            if (alp - 0 > epsilon and 1 - alp > epsilon):
                KM = gaussianKM (X[idx], X[idx], gamma)
                AlphaY = np.multiply (alphas, Y)
                AlphaY = np.repeat(AlphaY, 1, axis=1)
                KMalphaY = np.multiply (KM, AlphaY)
                KMalphaY = np.sum(KMalphaY, axis=0)
                b = float (Y[idx, 0] - KMalphaY)
                return b
    
    b = findBias ()

    # Saving the model
    model = (alphas, b)
    modelfile = 'gaussianBinary.pickle' 
    with open(modelfile, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gaussianKM (X, P, gamma):
    """
    Find the Kernel Matrix KM
    """
    X = np.matrix(X)
    P = np.matrix(P)
    m, n = X.shape
    p, n = P.shape

    XPt = X * P.T
    D1 = np.diag(X * X.T).reshape(m, 1)
    D1 = np.matrix(np.repeat(D1, p, axis=1))
    D2 = np.diag(P * P.T).reshape(p, 1)
    D2 = np.matrix(np.repeat(D2, m, axis=1))
    D2 = D2.T

    KM = D1 + D2 - 2 * XPt
    return np.exp (-gamma * KM)

def predict (X, Y, testX, testY, gamma=0.05):
    testX = np.matrix(testX)
    testY = np.matrix(testY)
    X = np.matrix(X)
    Y = np.matrix(Y).T
    p, n = testX.shape

    # Load the model
    modelfile = 'gaussianBinary.pickle'
    with open(modelfile, 'rb') as handle:
            (alphas, b) = pickle.load(handle)

    AlphaY = np.multiply (alphas, Y)
    AlphaY = np.repeat(AlphaY, p, axis=1)

    KM = gaussianKM (X, testX, gamma)
    KMalphaY = np.multiply (KM, AlphaY)
    KMalphaY = np.sum(KMalphaY, axis=0) + b

    predictions = np.array(np.sign(KMalphaY))[0,:]
    return predictions


def findAccuracy (predictions, Y):
    Y = np.array(Y)
    accuracy = float (np.sum(predictions == Y)) / len (Y)
    return accuracy