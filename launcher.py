# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'launcher.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from window_coherence import Ui_window_Coherence
import window_classes

class Ui_window_Launcher(object):
    def openCoherenceWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_window_Coherence()
        self.ui.setupUi(self.window)
        self.window.show()

    def openEpochsWindow(self):
        w = window_classes.MyEpochsForm()
        w.show

    def setupUi(self, window_Launcher):
        window_Launcher.setObjectName("window_Launcher")
        window_Launcher.resize(510, 361)
        self.centralwidget = QtWidgets.QWidget(window_Launcher)
        self.centralwidget.setObjectName("centralwidget")
        self.but_epochs = QtWidgets.QPushButton(self.centralwidget)
        self.but_epochs.setGeometry(QtCore.QRect(80, 40, 121, 41))
        self.but_epochs.setObjectName("but_epochs")

        self.but_epochs.clicked.connect(self.openEpochsWindow)


        self.but_coherence = QtWidgets.QPushButton(self.centralwidget)
        self.but_coherence.setGeometry(QtCore.QRect(290, 40, 141, 41))
        self.but_coherence.setObjectName("but_coherence")

        self.but_coherence.clicked.connect(self.openCoherenceWindow)

        window_Launcher.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(window_Launcher)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 510, 31))
        self.menubar.setObjectName("menubar")
        window_Launcher.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(window_Launcher)
        self.statusbar.setObjectName("statusbar")
        window_Launcher.setStatusBar(self.statusbar)

        self.retranslateUi(window_Launcher)
        QtCore.QMetaObject.connectSlotsByName(window_Launcher)

    def retranslateUi(self, window_Launcher):
        _translate = QtCore.QCoreApplication.translate
        window_Launcher.setWindowTitle(_translate("window_Launcher", "MainWindow"))
        self.but_epochs.setText(_translate("window_Launcher", "Epochs Analysis"))
        self.but_coherence.setText(_translate("window_Launcher", "Coherence"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window_Launcher = QtWidgets.QMainWindow()
    ui = Ui_window_Launcher()
    ui.setupUi(window_Launcher)
    window_Launcher.show()
    sys.exit(app.exec_())

