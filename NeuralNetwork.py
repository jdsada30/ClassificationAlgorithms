#!/usr/bin/env python
# coding: utf-8

#@author Hercules Hjalmarsson

from statistics import mean
from sklearn.base import BaseEstimator
import math
import numpy as np
import pandas as pd
import scipy
import scipy.stats

class neuralNetwork(BaseEstimator):
    def __init__(self, hiddenLayerSize, iterations=10000, showLoss = False,overFitPrevention = False):
        self.iter = iterations
        self.showLossStatus = showLoss
        self.overFitP = overFitPrevention
        self.hls = hiddenLayerSize
    
    #Function to define the size of the layer
    def layer_sizes(self,X, Y):
        n_x = X.shape[0] # size of input layer
        n_h = self.hls# size of hidden layer
        n_y = Y.shape[0] # size of output layer
        return (n_x, n_h, n_y)
    
    #Math helper functions
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    
    def sigmoidPrime(self, s):
        return s * (1 - s)
        
    def softmax(self,inputData):
        exps = np.exp(inputData - np.max(inputData, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)
    
    #Computes the loss value
    def cross_entropy(self,pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples
    
    def fit(self,X,Y,LearnRate=0.5):
        # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
        self.n_x = self.layer_sizes(X, Y, self.hls)[0]
        self.n_y = self.layer_sizes(X, Y, self.hls)[2]

        #Stores the target value for accuracy comparison when evaluating
        self.targetData = Y
    
        self.W1 = np.random.randn(self.hls, self.n_x) * 0.01 #weight matrix of shape (n_h, n_x)
        self.b1 = np.zeros(shape=(self.hls, 1))  #bias vector of shape (n_h, 1)
        self.W2 = np.random.randn(self.n_y, self.hls) * 0.01   #weight matrix of shape (n_y, n_h)
        self.b2 = np.zeros(shape=(self.n_y, 1))  #bias vector of shape (n_y, 1)
        self.ConstantW = np.random.randn(self.hls, self.n_x) * 0.01
        self.internalModel = { 'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        self.trainNeuralNetwork(LearnRate,X,Y)
             
    def trainNeuralNetwork(self,LearnRate,X,Y):
        # Loop (gradient descent)
        for i in range(self.iter):

            # Forward propagation. Inputs: "parameters, X". Outputs: "A2"
            o = self.forwardPropagate(X)
            self.predictTargetVal = self.formatOutput(o)
            
            #Gets the loss value
            self.loss = self.cross_entropy(self.A2,Y)

            # Backpropagation. Inputs: "parameters, X, Y".
            self.backPropagate(X,Y)
            
            # Train the weights and biases Inputs: "parameters, learnRate"
            self.trainWeightsAndBiases(LearnRate)
            
            #Map of model
            self.internalModel = { 'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
            
            #If used, makes sure the training accuracy isn't 1 for multiple iterations which may cause overfitting
            if(self.overFitP):
                if(self.evaluate(X,Y,True) >= 0.99):
                    print("Overfit detected! Exiting training loop.")
                    print("Error Rate: %f Accuracy: %f" % (self.loss,self.evaluate(X,Y,True)))
                    break
            # Print the loss value every 100 iterations
            if(self.showLossStatus and (i % 100) == 0):
                print(" Loss: %f Accuracy: %f" % (self.loss.sum(),self.evaluate(X,Y,True)))
                
    # Gradient descent parameter update
    def trainWeightsAndBiases(self,learningRate):
        # Update rule for each parameter
        self.W1 = self.W1 - learningRate * self.dW1
        self.W2 = self.W2 - learningRate * self.dW2
        self.b1 = self.b1 - learningRate * self.db1
        self.b2 = self.b2 - learningRate * self.db2
    
    #Predict function which outputs the model predicted values
    def predict(self,testData):
        
        outputSize = testData.shape[0]
        shouldCorrectOutput = False
        if(testData.shape[0] != self.W1.shape[1]):
            shouldCorrectOutput = True
            if(testData.shape[0] < self.W1.shape[1]):
                while testData.shape[0] < self.W1.shape[1]:
                    testData = testData.append([testData],ignore_index=True)

                for i in reversed(range(self.W1.shape[1])):
                    if(testData.shape[0] > self.W1.shape[1]):
                        testData = testData.drop([i])
            else:
                for i in reversed(range(self.W1.shape[1])):
                    if(testData.shape[0] > self.W1.shape[1]):
                        testData = testData.drop([i])

        self.A2 = self.forwardPropagate(testData)
        
        if(shouldCorrectOutput):
            for i in reversed(range(outputSize)):
                if(len(self.A2.shape) > outputSize):
                    self.A2 = np.delete(self.A2, i)
            return self.formatOutput(self.A2)
        else:
            return self.formatOutput(self.A2)
    
    #Forward propagation function to get a prediction by the using the models weights and biases
    def forwardPropagate(self,X):
        self.Z1 = np.add(np.matmul(self.W1, X), self.b1)
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.add(np.matmul(self.W2, self.A1), self.b2)
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    #Backwards propagation function to train the model with the predictions given by the forward propogate
    def backPropagate(self,X,Y):
        m = X.shape[1]
        dZ2 = self.A2 - Y
        self.dW2 = (1.0/m) * np.matmul(dZ2, np.transpose(self.A1)) + (0.7/m)*self.W2 #regularization
        self.db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)
    
        dZ1 = np.matmul(np.transpose(self.W2), dZ2) * (1 - np.power(self.A1, 2))
        self.dW1 = (1.0/m) * np.matmul(dZ1, np.transpose(X)) + (0.7/m)*self.W1 #regularization
        self.db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    #Functions which formats the output predictions correctly
    def formatOutput(self,output):
        getOutput = []
        for i in range(len(output)):
            getOutput.append(mean(output[i]))
        
        return getOutput
    
    #Function which evaluates the training and testing accuracy of the model
    def evaluate(self,testData,targetData,showTrainingAccuracy = False, showTestAccuracy = False):
        
        if(showTrainingAccuracy):
            temp = []
            accuracyT = 0
            for v in self.predictTargetVal: 
                if(v < 0.5):
                    temp.append(0)
                else:
                    temp.append(1)

            for j in range(len(self.targetData)):
                if(temp[j] == self.targetData[j]):
                    accuracyT += 1;
            
        if(showTestAccuracy):
            output = self.predict(testData)
            for i in range(len(output)):
                if(output[i] > 0.5):
                    output[i] = 1;
                else:
                    output[i] = 0;
            
            valid = targetData
            accuracy = 0 
            for i in range(len(valid)):
                if(output[i] == valid[i]):
                    accuracy += 1

        if(showTrainingAccuracy and showTestAccuracy):
            return accuracyT/len(self.targetData),accuracy/len(valid)
        
        if(showTrainingAccuracy):
            return accuracyT/len(self.targetData)
        
        if(showTestAccuracy):
            return accuracy/len(valid)

