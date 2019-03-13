import sys
import libsvm_model
import linear_model 
from read import get_data, filter_data

if __name__ == '__main__':
    if (len(sys.argv) < 5):
        print ("Go Away")
        exit(1)

    trainset = sys.argv[1]
    testset = sys.argv[2]
    binaryormulti = sys.argv[3]
    part = sys.argv[4]

    # Read Data
    trainData, testData = get_data (trainset, testset)
    # X, Y, testX, testY = get_data (trainset, testset)
    # print (X.shape, Y.shape, testX.shape, testY.shape)


    if (binaryormulti == '0'):
        # Binary classification
        classA, classB = 1, 2
        X, Y = filter_data (trainData, classA, classB)
        testX, testY = filter_data (testData, classA, classB)

        if (part == 'a'):
            linear_model.binary (X, Y, testX, testY)
        elif (part == 'c'):
            # libsvm_model.binary (X, Y, testX, testY)
            libsvm_model.binary (X, Y, testX, testY, kernel="gaussian")
        exit(0)

    else:
    # Multiclass classification
        if (part == 'b'):
            Data = []
            for i in range (10):
                Data.append([])
                for j in range (10):
                    Data[i].append(0)
                for j in range (i+1, 10):
                    X, Y = filter_data (trainData, classA=i, classB=j)
                    Data[i][j] = (X, Y)

            testX, testY = filter_data (testData, filter=False)
            libsvm_model.multi (Data, testX, testY)
        exit(0)
            


    print ("Go Away")