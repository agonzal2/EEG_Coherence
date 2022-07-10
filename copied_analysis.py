from copied_windows import Ui_w_perm_copy
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication, QFileDialog, QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QPushButton, QErrorMessage
from PyQt5.QtCore import pyqtSlot


import matplotlib.backends.backend_pdf




class MyForm(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_w_perm_copy()
    self.ui.setupUi(self)

    self.ui.ButtonSelectFolder.clicked.connect(self.selectFolder)
    self.ui.ButDelFile.clicked.connect(self.delFile)
    self.ui.ButtonRunAll.clicked.connect(self.runAll)
    
    self.freq_list_results = []

    # Error message (it is necessary to initialize it too)
    self.error_msg = QErrorMessage()
    self.error_msg.setWindowTitle("Error")
    self.show()


  def selectFolder(self):
    DataFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if DataFolder:
      self.ui.labelFileSelected.setText(DataFolder)
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def delFile(self):
    self.ui.labelFileSelected.clear()

  def runAll(self):
    a = 0

  

if __name__=="__main__":
     app = QApplication(sys.argv)
     w = MyForm()
     w.show

     sys.exit(app.exec())
