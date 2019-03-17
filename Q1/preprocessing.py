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
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk import everygrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Reading the data
def json_reader(fname, count=1000, stemming=False, bigrams=False):
    """
        Read multiple json files
        Args:
            fname: str: input file
        Returns:
            generator: iterator over documents 
    """
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    
    def convertAscii(text):
        return ''.join([i if ord(i) < 128 else '' for i in text])

    for line in open(fname, mode="r"):
        if (count <= 0):
            break
        count -= 1
        
        rating = re.search('stars": (.+?),', line)
        if rating:
            rating = int(float(rating.group(1)))
        else:
            print ('contin')
            continue
            
        review = re.search('"text": "(.+?)"', line)
        if review:
            review = review.group(1)
        else:
            print ('contin')
            continue

        review = np.array(word_tokenize(review))
        # print ('review', stemming)
        if (not stemming):
            # print ('stem')
            if bigrams:
                print('bi')
                if (len (review) < 3): # ignore it
                    continue
                review = np.array(list(everygrams(review, 1, 2)))
            # print ('yield')
            yield {'rating': rating, 'review': review}
        else:
            stopped_tokens = filter(lambda token: token not in en_stop, review)
            stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
            review = np.array(list(stemmed_tokens))
            if bigrams:
                review = np.array(list(everygrams(review, 1, 2)))
            yield {'rating': rating, 'review': review}       


def genVocab (trainset, vocabName, count=1000, size=100000, stemming=False, bigrams=False):
    """
    Load the data, make vocabulary of all unigrams
    """
    # Making of the dictionary
    tick = time.time()
    dictionary = {}
    for data in json_reader(trainset, count, stemming=stemming, bigrams=bigrams):
        for word in data['review']:
            if (word in dictionary):
                dictionary[word] += 1
            else:
                dictionary[word] = 1

    # Applying min_df and max_df
    print (len(dictionary))
    mins = []
    maxs = []
    for word, count in list(dictionary.items()):
        if (count < 5):
            mins.append(word)
            dictionary.pop(word, None)
        elif (count > 500):
            maxs.append(word)
            dictionary.pop(word, None)

    print (len(mins), len(maxs), len(dictionary))

    # Applying max_features
    dictionary = sorted(dictionary, key=dictionary.get, reverse=True)[:size]
            
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

    count = 200000
    if (sys.argv[1] == 'u'):
        # All Unigrams without stemming
        genVocab (sys.argv[2], 'Q1/vocabs/unigramVocab.pickle', count=count, size=200000)
    elif (sys.argv[1] == 's'):
        # All Unigrams with stemming
        genVocab (sys.argv[2], 'Q1/vocabs/stemmedVocab.pickle', count=count, size=200000, stemming=True)
    elif (sys.argv[1] == 'ub'):
        # All Bigrams
        genVocab (sys.argv[2], 'Q1/vocabs/bigramVocab.pickle', count=count, size=200000, bigrams=True)
    elif (sys.argv[1] == 'sb'):
        # All Bigrams with stemming
        genVocab (sys.argv[2], 'Q1/vocabs/stemmedVocab.pickle', count=count, size=200000, stemming=True)
    else:
        print ("Go Away")
