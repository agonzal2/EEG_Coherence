import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.weightstats import ttest_ind as wttest
import scipy.signal
import scipy.io
from scipy.signal import iirnotch
from scipy import spatial
from scipy import stats
import time
import struct
from copy import deepcopy
from indiv_calculations import *
import xlsxwriter

from w_record_viewer import *

from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication, QFileDialog, QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QPushButton, QErrorMessage
from PyQt5.QtCore import pyqtSlot


import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from datetime import datetime

import mne
#import xlrd

from pathlib import Path

from initial_processes import *

montage_name = 'standard_32grid_Alfredo'
#montage_name = 'standard_16grid_taini1'
#montage_name = '/media/jorge/DATADRIVE0/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'

sampling_rate = 1000
source_name = '100' # how do the name of the recording files begin

class MyForm(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_window_record_viewer()
    self.ui.setupUi(self)
    # Individual records
    self.ui.ButtonAdd_IndRecords.clicked.connect(self.add_IndRecords)
    self.ui.ButtonDelete_IndRecords.clicked.connect(self.del_IndRecords)
    self.ui.ButtonLoadRecordings.clicked.connect(self.load_recordings)
    self.ui.ButtonPlotRawData.clicked.connect(self.plotRawData)
    self.ui.ButtonPlotPS.clicked.connect(self.plotPS)
    self.ui.radioSelectAll.clicked.connect(self.selectAllElectrodes)
    self.ui.radioSelectNone.clicked.connect(self.selectNoElectrodes)
    self.ui.radioSelectLeftHem.clicked.connect(self.selectLeftHem)
    self.ui.radioSelectRightHem.clicked.connect(self.selectRightHem)
    self.ui.ButtonApplyBandFilter.clicked.connect(self.apply_band_filter)
    
    self.ui.ButtonCloseFigures.clicked.connect(self.closeFigures)
    #self.ui.ButtonExportIndPDF.clicked.connect(self.print2pdf)
    #self.ui.ButtonExportIndPNG.clicked.connect(self.print2png)
    # Recordings scroll
    self.ui.ScrollBarCurrentRecord.valueChanged.connect(self.changeRecording)
    # Variables
    self.recordings = [] # list of recordings that we could scroll down.
    self.currentRecording = 0

    # Error message (it is necessary to initialize it too)
    self.error_msg = QErrorMessage()
    self.error_msg.setWindowTitle("Error")
    self.show()


  def add_IndRecords(self):
    DataFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if DataFolder:
      self.ui.listWidget_Indiv_recordings.addItem(DataFolder)
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def del_IndRecords(self):
    self.ui.listWidget_Indiv_recordings.takeItem(self.ui.listWidget_Indiv_recordings.currentRow())

  def load_recordings(self):
    # selecting the montage
    if self.ui.tabWidget.currentIndex() == 0:
      rec_type = 'openeph'
    elif self.ui.tabWidget.currentIndex() == 1:
      rec_type = 'taini'
    # sampling rate of the recordings
    self.sampling_rate = self.ui.SpinBoxSamplingRate.value()

    print('loading recordings')
    for i in range(self.ui.listWidget_Indiv_recordings.count()):
      root_dir = str(self.ui.listWidget_Indiv_recordings.item(i).text())
      os.chdir(root_dir)
      # getting all the npy files in the folder
      d = os.getcwd() + '/'
      matching_files = glob.glob(r'*npy')
      for j, matching_file in enumerate(matching_files):
        new_recording = indiv_tests(root_dir + "/" + matching_file, i+j, self.sampling_rate)

        if rec_type == 'openeph':
          montage_name = '/media/jorge/DATADRIVE0/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'
          new_recording.load_npy32openephys(montage_name)
        elif rec_type == 'taini':
          montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
          new_recording.load_npy16taini(montage_name)
        
        ind_calc.append(new_recording)
        self.recordings.append(root_dir + "/" + matching_file)

    self.changeRecording()
    self.ui.ScrollBarCurrentRecord.setMaximum(len(self.recordings)-1)

  def changeRecording(self):
    self.currentRecording = self.ui.ScrollBarCurrentRecord.value()
    self.ui.labelCurrentRecording.setText(self.recordings[self.currentRecording])
    self.ui.label_Tmax.setText("(" + str(ind_calc[self.currentRecording].rawdata.n_times//1000) + " s)")
    self.ui.BoxTmax.setValue(ind_calc[self.currentRecording].rawdata.n_times//1000)
  
  def apply_band_filter(self):
    if self.ui.radioUseGPUfiltering.isChecked():
      n_jobs = 'cuda'
    else:
      n_jobs = self.ui.BoxFilterCores.value()
    
    self.checkElectrodes()
    # calls the class method to filter
    ind_calc[self.currentRecording].bandpass(self.ui.BoxLowFreq.value(), self.ui.BoxHighFreq.value(), 
                                self.electrodes, n_jobs)
    

  def plotRawData(self):
    self.checkElectrodes()
    ind_calc[self.currentRecording].plotRawData(self.ui.BoxBinSize.value(), self.ui.BoxTmin.value(), self.electrodes)

  def plotPS(self):
    self.checkElectrodes()
    ind_calc[self.currentRecording].plotPS(self.ui.BoxTmin.value(), self.ui.BoxTmax.value(), self.electrodes)

  def checkElectrodes(self):
    self.electrodes = []
    if self.ui.tabWidget.currentIndex() == 0:
      if self.ui.e_0.isChecked(): self.electrodes.append(0)
      if self.ui.e_1.isChecked(): self.electrodes.append(1)
      if self.ui.e_2.isChecked(): self.electrodes.append(2)
      if self.ui.e_3.isChecked(): self.electrodes.append(3)
      if self.ui.e_4.isChecked(): self.electrodes.append(4)
      if self.ui.e_5.isChecked(): self.electrodes.append(5)
      if self.ui.e_6.isChecked(): self.electrodes.append(6)
      if self.ui.e_7.isChecked(): self.electrodes.append(7)
      if self.ui.e_8.isChecked(): self.electrodes.append(8)
      if self.ui.e_9.isChecked(): self.electrodes.append(9)
      if self.ui.e_10.isChecked(): self.electrodes.append(10)
      if self.ui.e_11.isChecked(): self.electrodes.append(11)
      if self.ui.e_12.isChecked(): self.electrodes.append(12)
      if self.ui.e_13.isChecked(): self.electrodes.append(13)
      if self.ui.e_14.isChecked(): self.electrodes.append(14)
      if self.ui.e_15.isChecked(): self.electrodes.append(15)
      if self.ui.e_16.isChecked(): self.electrodes.append(16)
      if self.ui.e_17.isChecked(): self.electrodes.append(17)
      if self.ui.e_18.isChecked(): self.electrodes.append(18)
      if self.ui.e_19.isChecked(): self.electrodes.append(19)
      if self.ui.e_20.isChecked(): self.electrodes.append(20)
      if self.ui.e_21.isChecked(): self.electrodes.append(21)
      if self.ui.e_22.isChecked(): self.electrodes.append(22)
      if self.ui.e_23.isChecked(): self.electrodes.append(23)
      if self.ui.e_24.isChecked(): self.electrodes.append(24)
      if self.ui.e_25.isChecked(): self.electrodes.append(25)
      if self.ui.e_26.isChecked(): self.electrodes.append(26)
      if self.ui.e_27.isChecked(): self.electrodes.append(27)
      if self.ui.e_28.isChecked(): self.electrodes.append(28)
      if self.ui.e_29.isChecked(): self.electrodes.append(29)
      if self.ui.e_30.isChecked(): self.electrodes.append(30)
      if self.ui.e_31.isChecked(): self.electrodes.append(31)
    elif self.ui.tabWidget.currentIndex() == 1:
      if self.ui.et_0.isChecked(): self.electrodes.append(0)
      if self.ui.et_1.isChecked(): self.electrodes.append(1)
      if self.ui.et_2.isChecked(): self.electrodes.append(2)
      if self.ui.et_3.isChecked(): self.electrodes.append(3)
      if self.ui.et_4.isChecked(): self.electrodes.append(4)
      if self.ui.et_5.isChecked(): self.electrodes.append(5)
      if self.ui.et_6.isChecked(): self.electrodes.append(6)
      if self.ui.et_7.isChecked(): self.electrodes.append(7)
      if self.ui.et_8.isChecked(): self.electrodes.append(8)
      if self.ui.et_9.isChecked(): self.electrodes.append(9)
      if self.ui.et_10.isChecked(): self.electrodes.append(10)
      if self.ui.et_11.isChecked(): self.electrodes.append(11)
      if self.ui.et_12.isChecked(): self.electrodes.append(12)
      if self.ui.et_13.isChecked(): self.electrodes.append(13)
      if self.ui.et_14.isChecked(): self.electrodes.append(14)
      if self.ui.et_15.isChecked(): self.electrodes.append(15)

  def selectAllElectrodes(self):
    if self.ui.radioSelectAll.isChecked(): self.changeElectrodeValue(True, True)

  def selectNoElectrodes(self):
    if self.ui.radioSelectNone.isChecked(): self.changeElectrodeValue(False, False)

  def selectLeftHem(self):
    if self.ui.radioSelectLeftHem.isChecked: self.changeElectrodeValue(True, False)

  def selectRightHem(self):
    if self.ui.radioSelectRightHem.isChecked: self.changeElectrodeValue(False, True)

  def changeElectrodeValue(self, LeftHem = True, RightHem = True):
    # OpenEphys electrodes
    self.ui.e_0.setChecked(LeftHem)
    self.ui.e_1.setChecked(LeftHem)
    self.ui.e_2.setChecked(LeftHem)
    self.ui.e_3.setChecked(LeftHem)
    self.ui.e_4.setChecked(LeftHem)
    self.ui.e_5.setChecked(LeftHem)
    self.ui.e_6.setChecked(LeftHem)
    self.ui.e_7.setChecked(LeftHem)
    self.ui.e_8.setChecked(LeftHem)
    self.ui.e_9.setChecked(LeftHem)
    self.ui.e_10.setChecked(LeftHem)
    self.ui.e_11.setChecked(LeftHem)
    self.ui.e_12.setChecked(LeftHem)
    self.ui.e_13.setChecked(LeftHem)
    self.ui.e_14.setChecked(LeftHem)
    self.ui.e_15.setChecked(LeftHem)
    self.ui.e_16.setChecked(RightHem)
    self.ui.e_17.setChecked(RightHem)
    self.ui.e_18.setChecked(RightHem)
    self.ui.e_19.setChecked(RightHem)
    self.ui.e_20.setChecked(RightHem)
    self.ui.e_21.setChecked(RightHem)
    self.ui.e_22.setChecked(RightHem)
    self.ui.e_23.setChecked(RightHem)
    self.ui.e_24.setChecked(RightHem)
    self.ui.e_25.setChecked(RightHem)
    self.ui.e_26.setChecked(RightHem)
    self.ui.e_27.setChecked(RightHem)
    self.ui.e_28.setChecked(RightHem)
    self.ui.e_29.setChecked(RightHem)
    self.ui.e_30.setChecked(RightHem)
    self.ui.e_31.setChecked(RightHem)
    # Taini electrodes
    self.ui.et_0.setChecked(RightHem)
    self.ui.et_1.setChecked(RightHem)
    self.ui.et_2.setChecked(RightHem)
    self.ui.et_3.setChecked(RightHem)
    self.ui.et_4.setChecked(RightHem)
    self.ui.et_5.setChecked(RightHem)
    self.ui.et_6.setChecked(RightHem)
    self.ui.et_7.setChecked(LeftHem)
    self.ui.et_8.setChecked(LeftHem)
    self.ui.et_9.setChecked(LeftHem)
    self.ui.et_10.setChecked(LeftHem)
    self.ui.et_11.setChecked(LeftHem)
    self.ui.et_12.setChecked(LeftHem)
    self.ui.et_13.setChecked(LeftHem)
    self.ui.et_14.setChecked(RightHem) # emg
    self.ui.et_15.setChecked(LeftHem) # emg

  def closeFigures(self):
    plt.close('all')

  #def print2pdf(self, filename=""):
#
  #  if filename:
  #    filename2 = '/' + filename + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.pdf'
  #    pdf = matplotlib.backends.backend_pdf.PdfPages(my_coherence.ResultsFolder + filename2)
  #    figs = [plt.figure(n) for n in plt.get_fignums()]
  #    for fig in figs:
  #      fig.savefig(pdf, format='pdf')
  #    pdf.close()
  #  else:
  #    self.error_msg.showMessage("It is necessary to select a folder")
#
  #def print2png(self):
  #  my_coherence.figFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
  #  if my_coherence.figFolder:
  #    prefix = '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
  #    for i in plt.get_fignums():
  #      plt.figure(i)
  #      plt.savefig(my_coherence.figFolder + prefix +'figure%d.png' % i)
  #  else:
  #    self.error_msg.showMessage("It is necessary to select a folder")



ind_calc = [] # list of indiv_tests classes

if __name__=="__main__":
     app = QApplication(sys.argv)
     w = MyForm()
     w.show

     sys.exit(app.exec())
