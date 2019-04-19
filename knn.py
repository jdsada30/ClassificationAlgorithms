#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:00:56 2019

@author: juandiaz
"""

import numpy as np
import pandas as pd
import math 
import scipy
import scipy.stats
import scipy.spatial
import operator


class KNearestNeighborClassifier:
    def __init__(self):
        self.k = 3
        self.trainingSet = None
        self.y_train = None
        
    def fit(self, xvectors, yvectors):
        self.trainingSet = xvectors
        self.y_train = yvectors
        self.k = int(math.sqrt(len(yvectors)))

    def getNeighbors(self,  testInstance):
        #testVector = np.array(testInstance)
        distances = []
        #length = len(testInstance)-1
        for x in range(len(self.trainingSet)):
            dist = scipy.spatial.distance.euclidean(np.array(testInstance), np.array(self.trainingSet.iloc[x]))
            distances.append((self.trainingSet.iloc[x], dist))
            distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][0])
        return neighbors

    def predictLabel(self, neighbors):
        d={}
        for neighbor in neighbors:
            if self.y_train.loc[neighbor.name] in d:
                d[self.y_train.loc[neighbor.name]] +=1
            else:
                d[self.y_train.loc[neighbor.name]] = 1 
   
        return max(d.items(), key=operator.itemgetter(1))[0]

    def predict(self,vectors):   
        predictions = []
        vectors = np.array(vectors)
        for i in range(len(vectors)):
            predictions.append(self.predictLabel(self.getNeighbors(vectors[i])))
        return predictions
    def score(self, X_test, y_test):
        prs = self.predict(X_test)
        testActualLabels = np.array(y_test)
        predictions = np.array(prs)
        correct = 0
        for x in range(len(predictions)):
            if testActualLabels[x] == predictions[x]:
                correct += 1
        return (correct/float(len(predictions))) 