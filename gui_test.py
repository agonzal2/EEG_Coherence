import sys 
from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication, QPlainTextEdit, QPushButton, QTextEdit
from PyQt5 import QtWidgets, uic #sys allows application to accept command line argumetns, but also ensures clean close of application
from PyQt5.QtCore import pyqtSlot
import os
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import decimate
import sys
sys.path.insert(1, '/home/melissa/Documents/EEG_Coherence-master/EEG_Coherence-master/')
from initial_processes import taininumpy2mnechannels
import parameters
prm = parameters.Parameters()

#copy functions from load_npytaini file to plot raw_data
npy_file = '/home/melissa/CDKL5_1960/CDKL51960.npy'
montage_name = '/home/melissa/Documents/EEG_Coherence-master/EEG_Coherence-master/standard_4grid_taini1.elc'
channels_list = [2,11,14,15]
prm.set_sampling_rate(250.41) # data downsampled by 8
sample_rate = prm.get_sampling_rate()
raw_data = taininumpy2mnechannels(npy_file, montage_name, sample_rate/2, channels_list)

#from create_epochs import analysis_times
#number of epochs 
#start_time=np.array([17082289])
#end_time=np.array([38716848])
# number_epochs = np.round((end_time - start_time)/5)*250.41
#print(number_epochs)

#make a numpy array of zeros for epochs
analysis_times=np.array([])

os.chdir('/home/melissa/gui/')
main_window_file = ('/home/melissa/gui/main_window.ui')


class Ui(QtWidgets.QMainWindow):
        def __init__(self):
            super(Ui,self).__init__()
            uic.loadUi('main_window.ui',self) #load the .ui file 
            self.button1 = QPushButton('1', self)
            self.button2 = QPushButton('2', self)
            self.button3 = QPushButton('3', self)
            self.button4 = QPushButton('4', self)

            self.no1 = QPlainTextEdit(self)
            self.no2 = QPlainTextEdit(self)
            self.no3 = QPlainTextEdit(self)
            #self.show(Ui) #show the GUI

            self.button1.clicked.connect(self.on_click_1)
            self.button2.clicked.connect(self.on_click_2)
            self.button3.clicked.connect(self.on_click_3)
            self.button4.clicked.connect(self.on_click_4)

        @pyqtSlot()
        def on_click_1(self):
            analysis_times.append('1') 
            self.no2.insertPlainText("1") #change these functions to save data to table 
        def on_click_2(self):
            analysis_times.append('2')
            self.no2.insertPlainText("2")
        def on_click_3(self):
            analysis_times.append('3') 
            self.no2.insertPlainText("3")
        def on_click_4(self):
            analysis_times.append('4') 
            self.no2.insertPlainText("4")      


if __name__=="__main__":
    app=QApplication(sys.argv) #create an instance of QtWidgets.QApplication
    window=Ui() #create an instance of our class
    window.show()
    raw_data.plot(scalings = "auto")
    plt.show()
    print(analysis_times)
    sys.exit(app.exec_()) #start the application


#Convert numpy array to excel spreadsheet
os.chdir('/home/melissa/Results')
df = pd.analysis_times()
filepath = 'analysis_times.xlsx'
df.to_excel(filepath, index=False)