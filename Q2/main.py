import sys
from libsvm_model import *
from read import get_data

if __name__ == '__main__':
    if (len(sys.argv) < 5):
        print ("Go Away")
        exit(1)

    trainset = sys.argv[1]
    testset = sys.argv[2]
    binaryormulti = sys.argv[3]
    part = sys.argv[4]

    # Read Data
    X, Y, testX, testY = get_data (trainset, testset)
    print (X.shape, Y.shape, testX.shape, testY.shape)


    if (binaryormulti == '0'):
        # Binary classification
        if (part == 'c'):
            binary (X, Y, testX, testY)
            binary (X, Y, testX, testY, kernel="gaussian")
        exit(0)

    print ("Go Away")