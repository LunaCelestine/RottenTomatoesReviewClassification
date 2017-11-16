# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:08:40 2017

@author: Bradley Dabdoub
"""


import numpy
import pandas
from sklearn.cross_validation import train_test_split 
from collections import Counter
import re
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

df1 = pandas.read_csv('neg.txt', delimiter='~',encoding ='latin1')
df1["label"] = -1
negReviews = df1.iloc[:, 0].values
#print(negReviews)
df2 = pandas.read_csv('pos.txt', delimiter='~',encoding ='latin1')
df2["label"] = 1
posReviews = df2.iloc[:, 0].values
#print(posReviews)

negY = df1.iloc[:,1].values
posY = df2.iloc[:,1].values

X_train_neg, X_test_neg, Y_train_neg, Y_test_neg = train_test_split(negReviews, negY, test_size=.3, random_state=1)
X_validation_neg, X_test_neg, Y_validation_neg, Y_test_neg = train_test_split(X_test_neg, Y_test_neg, test_size=.5, random_state=1)


X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(posReviews, posY, test_size=.3, random_state=1)
X_validation_pos, X_test_pos, Y_validation_pos, Y_test_pos = train_test_split(X_test_pos, Y_test_pos, test_size=.5, random_state=1)

X_train = numpy.concatenate([X_train_neg, X_train_pos])
Y_train = numpy.concatenate([Y_train_neg, Y_train_pos])
X_validate = numpy.concatenate([X_validation_neg, X_validation_pos])
Y_validate = numpy.concatenate([Y_validation_neg, Y_validation_pos])
X_test = numpy.concatenate([X_test_neg, X_test_pos])
Y_test = numpy.concatenate([Y_test_neg, Y_test_pos])

def bagOWords(reviews, labels, target):
    wordList = []
    for index,review in enumerate(reviews):
        if (labels[index] == target):
            reviewWords = review.split()
            for reviewWord in reviewWords:
                wordList.append(reviewWord)
    return wordList
    
posBag = bagOWords(X_train, Y_train, 1)
negBag = bagOWords(X_train, Y_train, -1)

negCounter = Counter(negBag)
posCounter = Counter(posBag)


#print(posCounter)
#print(negCounter == posCounter)

labelCount = Counter(Y_train)
negCount = labelCount.get(-1)
posCount = labelCount.get(1)

negClassProb = negCount / (negCount + posCount)
posClassProb = posCount / (negCount + posCount)
#print(negClassProb)
#print(posClassProb)



def NaiveBayes(review, count, prob):
    pred = 1
    reviewWords = Counter(re.split("\s+", review))
    for reviewWord in reviewWords:
        pred *= reviewWords.get(reviewWord) * ((count.get(reviewWord, 0) + 1) / sum(count.values()))
    pred *= prob
    return pred  
 
def predict(review):
    negativePrediction = NaiveBayes(review, negCounter, negClassProb)
    positivePrediction = NaiveBayes(review, posCounter, posClassProb)
    
    
    if negativePrediction > positivePrediction:
        return -1
    return 1
def main():
    predictions = []
    for review in X_test:
        predictions.append(predict(review))    
    

    print('Misclassified samples: %d' % (Y_test != predictions).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, predictions))
    print('F1 Score: ' +  str(f1_score(Y_test, predictions)))
main()





            
    





















