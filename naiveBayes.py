#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:56:06 2019

@author: juandiaz
"""
import numpy as np
import pandas as pd
import math 
import scipy
import scipy.stats

class NaiveBayesClassifier:
    
    def __init__(self):
        self.init = 1 
    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))
    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
        return math.sqrt(variance)
    def summarize(self,instances):
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*instances)]
        return summaries
    def calculateClassProbabilities(self,summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= scipy.stats.norm(mean, stdev).pdf(x)
        return probabilities
              
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.separatedBasedOnOutcomeClass()
        self.summaries = {}
        for classValue, instances in self.separated.items():
            self.summaries[classValue] = self.summarize(instances)
        print("Model Trained")
        
    def single_prediction(self, inputVector):
        probabilities = self.calculateClassProbabilities(self.summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def predict(self,X_test):
        predictions = []
        for i in range(len(X_test)):
            predictions.append(self.single_prediction( X_test.iloc[i]))
        return predictions
        
    def separatedBasedOnOutcomeClass(self):
        self.separated = {}
        class_vals = np.array(self.y_train.unique())
        for val in class_vals: 
            self.separated[val]= []
            i=0
            for outcome in np.array(self.y_train.where(self.y_train==val)):
                if outcome == val:
                    self.separated[val].append(self.X_train.iloc[i])
                i+=1
        return self.separated
    def score(self, X_test, y_test):
        prs = self.predict(X_test)
        testActualLabels = np.array(y_test)
        predictions = np.array(prs)
        correct = 0
        for x in range(len(predictions)):
            if testActualLabels[x] == predictions[x]:
                correct += 1
        return (correct/float(len(predictions))) 
        
    



