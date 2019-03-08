"""
Read the data
"""

import numpy as np

def read_data (datafile="../dataset/SVM/train.csv", entryno=1, binary=True):
    lines = np.array([(line.rstrip('\n')).split(',') for line in open(datafile)], dtype=np.int64)
    if binary:
        lines = lines[ (lines[:,-1] == entryno) | (lines[:,-1] == entryno+1) ]

    np.random.seed(0) # Deterministically random
    np.random.shuffle(lines)
    X = lines[:,:-1] / 255.0
    Y = lines[:,-1] * 2 - 3 # for having y = {-1, 1}
    
    return X, Y
    

def get_data (trainset, testset):
    X, Y = read_data(trainset)
    testX, testY = read_data(testset)
    return X, Y, testX, testY