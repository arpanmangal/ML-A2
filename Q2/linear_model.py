"""
Code for SVM with linear kernel
"""


import time
import numpy as np
import cvxopt as cvx


def binary (X, Y, testX, testY):
    """
    Train SVM with linear kernel
    """

    tick = time.time()

    X = np.matrix(X)
    Y = np.matrix(Y).T
    testX = np.matrix(testX)
    testY = np.matrix(testY).T
    m, n = X.shape

    # Parameters for CVXOPT
    KM = X * X.T
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

    # Finding Alphas
    alphas = np.array(sol['x'])
    print (alphas.shape)

    # Finding Weight matrix
    W = np.multiply(X, Y)
    W = np.multiply(W, alphas)
    W = W.sum(axis=0)
    W = np.array(W)[0, :]

    # Finding B
    def findB ():
        epsilon = 1e-5
        for idx, alp in enumerate(alphas):
            if (alp - 0 > epsilon and 1 - alp > epsilon):
                print (alp)
                y = float (Y[idx])
                w = np.matrix(W)
                x = np.matrix(X[idx, :]).T
                b = float (y - w*x)
                print (y, w.shape, x.shape, b)
                return b
        
    b = findB ()

    # Predictions
    def getPredAcc (X, Y):
        pred = X*W
        pred = np.array(pred)[:,0]
        values = np.sign(pred)
        Y = np.array(Y)[:,0]
        accuracy = np.sum(values == Y) / len(Y)
        return float(accuracy)

    W = np.matrix(W).T
    trainAccuracy = getPredAcc(X, Y)
    testAccuracy = getPredAcc(testX, testY)
    print ("Train Accuracy: ", trainAccuracy)
    print ("Test Accuracy: ", testAccuracy)

    # Support Vectors
    epsilon = 1e-5
    sv = []
    for idx, alp in enumerate(alphas):
        if (alp - 0 > epsilon and 1 - alp > epsilon):
            sv.append(alp)
    print ("Number of Support Vectors: ", len(sv))


    print ("Time taken: ", time.time() - tick)
    return testAccuracy

