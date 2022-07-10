# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'copied_windows.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_w_perm_copy(object):
    def setupUi(self, w_perm_copy):
        w_perm_copy.setObjectName("w_perm_copy")
        w_perm_copy.resize(643, 237)
        self.centralwidget = QtWidgets.QWidget(w_perm_copy)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 931, 761))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.ButtonRunAll = QtWidgets.QPushButton(self.tab)
        self.ButtonRunAll.setGeometry(QtCore.QRect(220, 116, 70, 51))
        self.ButtonRunAll.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.ButtonRunAll.setObjectName("ButtonRunAll")
        self.frame_Experimental = QtWidgets.QFrame(self.tab)
        self.frame_Experimental.setGeometry(QtCore.QRect(10, 20, 611, 71))
        self.frame_Experimental.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_Experimental.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_Experimental.setObjectName("frame_Experimental")
        self.ButtonSelectFolder = QtWidgets.QPushButton(self.frame_Experimental)
        self.ButtonSelectFolder.setGeometry(QtCore.QRect(9, 19, 81, 31))
        self.ButtonSelectFolder.setObjectName("ButtonSelectFolder")
        self.ButDelFile = QtWidgets.QPushButton(self.frame_Experimental)
        self.ButDelFile.setGeometry(QtCore.QRect(550, 20, 51, 31))
        self.ButDelFile.setObjectName("ButDelFile")
        self.labelFileSelected = QtWidgets.QLabel(self.frame_Experimental)
        self.labelFileSelected.setGeometry(QtCore.QRect(100, 19, 441, 31))
        self.labelFileSelected.setText("")
        self.labelFileSelected.setObjectName("labelFileSelected")
        self.tabWidget.addTab(self.tab, "")
        w_perm_copy.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(w_perm_copy)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 643, 20))
        self.menubar.setObjectName("menubar")
        w_perm_copy.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(w_perm_copy)
        self.statusbar.setObjectName("statusbar")
        w_perm_copy.setStatusBar(self.statusbar)

        self.retranslateUi(w_perm_copy)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(w_perm_copy)

    def retranslateUi(self, w_perm_copy):
        _translate = QtCore.QCoreApplication.translate
        w_perm_copy.setWindowTitle(_translate("w_perm_copy", "MainWindow"))
        self.ButtonRunAll.setText(_translate("w_perm_copy", "RUN"))
        self.ButtonSelectFolder.setText(_translate("w_perm_copy", "Select folder"))
        self.ButDelFile.setText(_translate("w_perm_copy", "Delete"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("w_perm_copy", "Permutation Copy"))

