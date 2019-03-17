#!/bin/bash

# Checking number of arguments
if [ "$#" -lt 4 ]; then
    echo "Invalid Arguments"
    exit 1
fi

if [ $1 == "v" ]; then
    # Generate vocabulary
    python3 Q1/preprocessing.py u dataset/NB/train.json
    # python3 Q1/preprocessing.py s dataset/NB/train.json
    # python3 Q1/preprocessing.py ub dataset/NB/train.json
    exit 0
elif [ "$1" -eq "1" ]; then
    python3 Q1/model.py $2 $3 $4
    exit 0
elif [ "$1" -eq "2" ]; then
    python3 Q2/main.py $2 $3 $4 $5
    exit 0
fi

echo "Invalid Arguments"
exit 1