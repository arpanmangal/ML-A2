# ML-A2
Assignment 2 -- Machine Learning

## Naive Bayes

## Setup
Generate the vocabularies using:
```
$ ./run.sh v 1 2 3
```

### Running
```
$ ./run.sh 1 dataset/NB/train.json dataset/NB/test.json <a-f>
$ ./run.sh 1 dataset/NB/train_full.json dataset/NB/test_full.json g
```

# SVM
## Setup
```
$ cd Q2
$ wget http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip
$ mv ...+zip libsvm.zip
$ unzip libsvm.zip
$ mv libsvm<version> Q2/libsvm
$ cd libsvm
$ make
$ cd python
$ make
```

## Running
Binary Classification
```
$ ./run.sh 2 dataset/SVM/train.csv dataset/SVM/test.csv 0 <<a-c>>
```

Multiclass Classification
```
$ ./run.sh 2 dataset/SVM/train.csv dataset/SVM/test.csv 1 <<a-d>>
```