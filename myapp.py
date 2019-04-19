# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

#MyApp.py
# D. Thiebaut
# PyQt5 Application
# Editable UI version of the MVC application.
# Inherits from the Ui_MainWindow class defined in mainwindow.py.
# Provides functionality to the 3 interactive widgets (2 push-buttons,
# and 1 line-edit).
# The class maintains a reference to the model that implements the logic
# of the app.  The model is defined in class Model, in model.py.

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from mainwindow import Ui_MainWindow
import sys
from model import Model
import pandas as pd

class MainWindowUIClass( Ui_MainWindow ):
    def __init__( self ):
        '''Initialize the super class
        '''
        super().__init__()
        self.model = Model()
        self.Trained = False
        self.model.setModel(1)
        
    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi( MW )

        # close the lower part of the splitter to hide the 
        # debug window under normal operations
        #self.splitter.setSizes([300, 0])

    def debugPrint( self, msg ):
        '''Print the message in the text edit at the bottom of the
        horizontal splitter.
        '''
        self.debugTextBrowser.append( msg )

    def refreshAll( self ):
        '''
        Updates the widgets whenever an interaction happens.
        Typically some interaction takes place, the UI responds,
        and informs the model of the change.  Then this method
        is called, pulling from the model information that is
        updated in the GUI.
        '''
        print("hi")
        self.lineEdit_3.setText(self.model.getXFileName())
        self.lineEdit.setText(self.model.getYFileName())
        self.lineEdit_2.setText(self.model.getPredictFileName(  ))
        #self.lineEdit_3.setText( self.model.getFileName() )
        #self.textEdit.setText( self.model.getFileContents().to_string() )
    #slot
    def knnSlot( self ):
        self.model.setModel(3)
        print("knn")
    #slot
    def annSlot( self ):
        self.model.setModel(2)
        print("ann")
    #slot
    def naiveBayesSlot( self ):
        self.model.setModel(1)
        print("naive bayes")
    
    # slot
    def returnPressedSlot( self ):
        ''' Called when the user enters a string in the line edit and
        presses the ENTER key.
        '''
        fileName =  self.lineEdit_3.text()
        if self.model.isValid( fileName ):
            self.model.setFileName( self.lineEdit_3.text() )
            self.refreshAll()
        else:
            m = QtWidgets.QMessageBox()
            m.setText("Invalid file name!\n" + fileName )
            m.setIcon(QtWidgets.QMessageBox.Warning)
            m.setStandardButtons(QtWidgets.QMessageBox.Ok
                                 | QtWidgets.QMessageBox.Cancel)
            m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = m.exec_()
            self.lineEdit_3.setText( "" )
            self.refreshAll()
            self.debugPrint( "Invalid file specified: " + fileName  )

    # slot
    def runClassifierSlot( self ):
        ''' Called when the user presses the Write-Doc button.
        '''
        self.debugPrint("Running classifier!")
        if self.Trained == False:
            self.debugPrint("Classifier has not yet been trained")
        else:
            time = self.model.runClassifier()
            self.debugPrint("Classifier ran successfully!" )
            self.debugPrint("Class predictions are stored in file predictions.csv" )
            self.debugPrint((("Total running time: %.2f seconds")%time) )
        
    
    # slot
    def trainClassifierSlot( self ):
        ''' Called when the user presses the Write-Doc button.
        '''
        ret, time = self.model.trainClassifier()
        if ret == 0:
            self.textEdit.setText( "Error training classifier, please make sure data sets are in correct format" )
        else:
            s = "Classifier Trained with "+ str(ret) + " objects."
            r =  (("Training time: %.2f seconds") %time)
            sr = s+r
            
            self.Trained = True
            self.textEdit.setText(sr)
        
    # slot
    def browseSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            self.debugPrint( "setting file name: " + fileName )
            self.model.setFileName( fileName )
            self.refreshAll()
    # slot
    def browseXTrainSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            
            #self.debugPrint("X training file name: " + fileName )
            self.model.setXFileName( fileName ) #change to setXfileName
            self.refreshAll()
    # slot
    def browseYTrainSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            #self.debugPrint( "setting file name: " + fileName )
            self.model.setYFileName( fileName ) #change to setXfileName#change to setYfileName
            self.refreshAll()
            
    # slot
    def browsePredictFileSlot( self ):
        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        "All Files (*);;Python Files (*.py)",
                        options=options)
        if fileName:
            #self.debugPrint( "setting file name: " + fileName )
            self.model.setPredictFileName( fileName )#change to setPredictfileName
            self.refreshAll()

def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

main()

