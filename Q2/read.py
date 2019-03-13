"""
Read the data
"""

import numpy as np

def read_lines (datafile="../dataset/SVM/train.csv"):
    lines = np.array([(line.rstrip('\n')).split(',') for line in open(datafile)], dtype=np.int64)
    np.random.seed(0) # Deterministically random
    np.random.shuffle(lines)
    return lines


def get_data (trainset, testset):
    trainLines = read_lines (trainset)
    testLines = read_lines (testset)
    return trainLines, testLines

def filter_data (lines, classA=1, classB=2, filter=True):
    if filter:
        lines = lines[ (lines[:,-1] == classA) | (lines[:,-1] == classB) ]
    
    X = lines[:,:-1] / 255.0
    Y = lines[:,-1]

    if filter:
        def binaryFi (y):
            if (y == classA):
                return 1
            else:
                return -1
        vf = np.vectorize(binaryFi)
        Y = vf (Y)
    
    return X, Y
