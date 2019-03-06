"""
Module having functions for
1. Reading the data
2. Preprocessing like removing stop words and stemming
3. Making the dictionary
"""

import sys
import json
import time
import pickle
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Reading the data
def json_reader(fname, count=1000, stemming=False):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    
    for line in open(fname, mode="r"):
        if (count <= 0):
            break
        count -= 1
        
        data = json.loads(line)
        rating = int(data['stars'])
        review = np.array(word_tokenize(data['text']))
        if (not stemming):
            yield {'rating': rating, 'review': review}
        else:
            stopped_tokens = filter(lambda token: token not in en_stop, review)
            stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
            review = np.array(list(stemmed_tokens))
            yield {'rating': rating, 'review': review}        
        

def allUniVocab (trainset, vocabName, count=1000, stemming=False):
    """
    Load the data, make vocabulary of all unigrams
    """
    # Making of the dictionary
    tick = time.time()
    dictionary = {}
    for data in json_reader(trainset, count, stemming=stemming):
        for word in data['review']:
            dictionary[word] = 0
            
    dictionary = { word : idx for idx, word in enumerate(dictionary)}
    print (len(dictionary))

    with open(vocabName, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print ("Time Taken: ", time.time() - tick)
    

def loadVocab (vocabName):
    """
    Load the dictionary
    """
    with open(vocabName, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary


if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print ("Go Away")
        exit(1)

    if (sys.argv[1] == 'u'):
        # All Unigrams without stemming
        allUniVocab (sys.argv[2], 'unigramVocab.pickle', 100000, False)
    elif (sys.argv[1] == 's'):
        # All Unigrams with stemming
        allUniVocab (sys.argv[2], 'stemmedVocab.pickle', 100000, True)
    else:
        print ("Go Away")
