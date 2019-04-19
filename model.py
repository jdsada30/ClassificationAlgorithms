#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:03:07 2019

@author: juandiaz
"""

# model.py
# D. Thiebaut
# This is the model part of the Model-View-Controller
# The class holds the name of a text file and its contents.
# Both the name and the contents can be modified in the GUI
# and updated through methods of this model.
# 
import pandas as pd
from naiveBayes import NaiveBayesClassifier
from knn import KNearestNeighborClassifier
import time 
class Model:
    def __init__( self ):
        '''
        Initializes the two members the class holds:
        the file name and its contents.
        '''
        self.fileName = None
        self.xfileName = None
        self.yfileName = None
        self.predictFileName = None
        self.predictFileContents = pd.DataFrame()
        self.xfileContents = pd.DataFrame()
        self.yfileContents = pd.DataFrame()
        self.output = None
        self.model = None
        print("hi")
    def setModel(self, modelNo):
        if modelNo == 1:
            self.model = NaiveBayesClassifier()
        if modelNo == 2:
            print("model not yet imported")
        if modelNo == 3:
            self.model = KNearestNeighborClassifier()
            print("model is set to knn")
            

    def isValid( self, fileName ):
        '''
        returns True if the file exists and can be
        opened.  Returns False otherwise.
        '''
        try: 
            file = open( fileName, 'r' )
            file.close()
            return True
        except:
            return False

    def setFileName( self, fileName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( fileName ):
            self.fileName = fileName
            self.fileContents = pd.read_csv(fileName) #open( fileName, 'r' ).read()
        else:
            self.fileContents = ""
            self.fileName = ""
    
    def setXFileName( self, fileName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( fileName ):
            self.xfileName = fileName
            self.xfileContents = pd.read_csv(fileName) #open( fileName, 'r' ).read()
        else:
            self.xfileContents = ""
            self.xfileName = ""
            
    
            
    def setYFileName( self, fileName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( fileName ):
            self.yfileName = fileName
            self.yfileContents = pd.read_csv(fileName) #open( fileName, 'r' ).read()
            self.yfileContents.columns = [0]
            self.yfileContents = self.yfileContents[0]
        else:
            self.yfileContents = ""
            self.yfileName = ""
    def setPredictFileName( self, fileName ):
        '''
        sets the member fileName to the value of the argument
        if the file exists.  Otherwise resets both the filename
        and file contents members.
        '''
        if self.isValid( fileName ):
            self.predictFileName = fileName
            self.predictFileContents = pd.read_csv(fileName) #open( fileName, 'r' ).read()
        else:
            self.xfileContents = ""
            self.xfileName = ""
    
    def getPredictFileName( self ):
        '''
        Returns the name of the file name member.
        '''
        return self.predictFileName        
    def getXFileName( self ):
        '''
        Returns the name of the file name member.
        '''
        return self.xfileName
    
    def getYFileName( self ):
        '''
        Returns the name of the file name member.
        '''
        return self.yfileName

    def getFileContents( self ):
        '''
        Returns the contents of the file if it exists, otherwise
        returns an empty string.
        '''
        return self.fileContents
    
    def trainClassifier( self ):
        '''
        Writes the string that is passed as argument to a
        a text file with name equal to the name of the file
        that was read, plus the suffix ".bak"
        '''
        if self.xfileContents is not None and self.yfileContents is not None:
            #print(self.xfileContents)
            #print(self.yfileContents)
            start = time.time()
            self.model.fit(self.xfileContents, self.yfileContents )
            end = time.time()
            elapsedTime = end - start
            return len(self.xfileContents), elapsedTime
        else:
            return 0, 0
            
    def runClassifier( self ):
        '''
        Writes the string that is passed as argument to a
        a text file with name equal to the name of the file
        that was read, plus the suffix ".bak"
        '''
        if self.isValid( self.predictFileName):
            start = time.time()
            predictions = self.model.predict(self.predictFileContents)
            out = pd.DataFrame({"Predictions":predictions})
            fileName =  "predictions.csv"
            out.to_csv(fileName,index=False)
            end = time.time()
            elapsedTime = end - start
            return elapsedTime
            #file = open( fileName, 'w' )
            #file.write(text)
            #file.close()