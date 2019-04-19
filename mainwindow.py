# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot

class Ui_MainWindow( QObject ):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.debugTextBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.debugTextBrowser.setObjectName("debugTextBrowser")
        self.gridLayout_2.addWidget(self.debugTextBrowser, 5, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 3, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setMaximumSize(QtCore.QSize(130, 16777215))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.radioButton = QtWidgets.QRadioButton(self.frame_2)
        self.radioButton.setObjectName("radioButton")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton)
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.frame_2)
        self.radioButton_2.setObjectName("radioButton_2")
        self.buttonGroup.addButton(self.radioButton_2)
        self.verticalLayout.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.frame_2)
        self.radioButton_3.setObjectName("radioButton_3")
        self.buttonGroup.addButton(self.radioButton_3)
        self.verticalLayout.addWidget(self.radioButton_3)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.gridLayout_2.addWidget(self.frame_2, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout.addWidget(self.lineEdit_2)
        self.PredictBrowse = QtWidgets.QPushButton(self.centralwidget)
        self.PredictBrowse.setObjectName("PredictBrowse")
        self.horizontalLayout.addWidget(self.PredictBrowse)
        self.gridLayout_2.addLayout(self.horizontalLayout, 4, 0, 1, 1)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.trainClassifier = QtWidgets.QPushButton(self.frame)
        self.trainClassifier.setObjectName("trainClassifier")
        self.gridLayout.addWidget(self.trainClassifier, 2, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_2.addWidget(self.lineEdit_3)
        self.XtrainBrowse = QtWidgets.QPushButton(self.frame)
        self.XtrainBrowse.setObjectName("XtrainBrowse")
        self.horizontalLayout_2.addWidget(self.XtrainBrowse)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.YtrainBrowse = QtWidgets.QPushButton(self.frame)
        self.YtrainBrowse.setObjectName("YtrainBrowse")
        self.horizontalLayout_2.addWidget(self.YtrainBrowse)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 2)
        self.textEdit = QtWidgets.QTextEdit(self.frame)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 1, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(623, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 2, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.runClassifier = QtWidgets.QPushButton(self.centralwidget)
        self.runClassifier.setObjectName("runClassifier")
        self.horizontalLayout_3.addWidget(self.runClassifier)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 6, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.XtrainBrowse.clicked.connect(self.browseXTrainSlot)
        self.trainClassifier.clicked.connect(self.trainClassifierSlot)
        self.YtrainBrowse.clicked.connect(self.browseYTrainSlot)
        self.PredictBrowse.clicked.connect(self.browsePredictFileSlot)
        self.runClassifier.clicked.connect(self.runClassifierSlot)
        self.radioButton.toggled['bool'].connect(self.naiveBayesSlot)
        self.radioButton_2.toggled['bool'].connect(self.annSlot)
        self.radioButton_3.toggled['bool'].connect(self.knnSlot)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_5.setText(_translate("MainWindow", "Train Classifier: Choose training files"))
        self.label_6.setText(_translate("MainWindow", "Predict Values: Choose a file to run classier and predict class values"))
        self.label.setText(_translate("MainWindow", "Classifier"))
        self.radioButton.setText(_translate("MainWindow", "Naive Bayes"))
        self.radioButton_2.setText(_translate("MainWindow", "Neural Network"))
        self.radioButton_3.setText(_translate("MainWindow", "K Nearest Neighbors"))
        self.label_3.setText(_translate("MainWindow", "File Name: "))
        self.PredictBrowse.setText(_translate("MainWindow", "Browse"))
        self.trainClassifier.setText(_translate("MainWindow", "Train Classifier"))
        self.label_4.setText(_translate("MainWindow", "X_Train File:"))
        self.XtrainBrowse.setText(_translate("MainWindow", "Browse"))
        self.label_2.setText(_translate("MainWindow", "Y_train File:"))
        self.YtrainBrowse.setText(_translate("MainWindow", "Browse"))
        self.runClassifier.setText(_translate("MainWindow", "Run Classifier"))
    
    
    @pyqtSlot( )
    def annSlot( self ):
        pass
    @pyqtSlot( )
    def naiveBayesSlot( self ):
        pass
    @pyqtSlot( )
    def knnSlot( self ):
        pass
    
    @pyqtSlot( )
    def runClassifierSlot( self ):
        pass
    @pyqtSlot( )
    def trainClassifierSlot( self ):
        pass
    @pyqtSlot( )
    def browsePredictFileSlot( self ):
        pass
    @pyqtSlot( )
    def browseXTrainSlot( self ):
        pass
    @pyqtSlot( )
    def browseYTrainSlot( self ):
        pass

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

