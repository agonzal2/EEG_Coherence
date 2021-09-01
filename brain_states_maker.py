import os
import glob
from w_brain_states import Ui_window_brain_states_maker
import numpy as np
import pandas as pd
import scipy.signal
import scipy.io
import time
import struct
from copy import deepcopy
from indiv_calculations import *
import xlsxwriter

from w_brain_states import *

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
    self.ui = Ui_window_brain_states_maker()
    self.ui.setupUi(self)
    # Individual records
    self.ui.ButtonAdd_IndRecords.clicked.connect(self.add_IndRecords)
    self.ui.ButtonDelete_IndRecords.clicked.connect(self.del_IndRecords)
    self.ui.ButtonLoadRecordings.clicked.connect(self.load_recordings)
    self.ui.ScrollBar_binchange.valueChanged.connect(self.changebin)
    # Brain States
    self.ui.Button_Wake.clicked.connect(self.addWake)
    self.ui.Button_NoREM.clicked.connect(self.addNoREM)
    self.ui.Button_REM.clicked.connect(self.addREM)
    self.ui.Button_other.clicked.connect(self.addOther)
    self.ui.Button_Seizure.clicked.connect(self.addSeizure)
    #
    self.ui.Button_ExportExcel.clicked.connect(self.export2excel)    

    self.ui.radioSelectAll.clicked.connect(self.selectAllElectrodes)
    self.ui.radioSelectNone.clicked.connect(self.selectNoElectrodes)
    self.ui.radioSelectLeftHem.clicked.connect(self.selectLeftHem)
    self.ui.radioSelectRightHem.clicked.connect(self.selectRightHem)
    
    #self.ui.ButtonExportIndPDF.clicked.connect(self.print2pdf)
    #self.ui.ButtonExportIndPNG.clicked.connect(self.print2png)
    # Recordings scroll
    self.ui.ScrollBarCurrentRecord.valueChanged.connect(self.changeRecording)
    # Variables
    self.recordings = [] # list of recordings that we could scroll down.
    self.file_names = [] # list of file names, without the folder path
    self.currentRecording = 0
    self.ResultsFolder = ''

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
    elif self.ui.tabWidget.currentIndex() == 2:
      rec_type = 'tetrodes'
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
          montage_name = '/media/jorge/otherprojects/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'
          new_recording.load_npy32openephys(montage_name)
        elif rec_type == 'taini':
          montage_name = '/media/jorge/otherprojects/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
          new_recording.load_npy16taini(montage_name)
        elif rec_type == 'tetrodes':
          montage_name = '/media/jorge/otherprojects/Code/coherence/EEG_Coherence/standard_16tetrodes.elc'
          new_recording.load_npy16tetrodes(montage_name)

        
        ind_calc.append(new_recording)
        self.recordings.append(root_dir + "/" + matching_file)
        self.file_names.append(matching_file)

    self.changeRecording()
    self.ui.ScrollBarCurrentRecord.setMaximum(len(self.recordings)-1)
  
  def changebin(self):
    self.current_sample = int(self.ui.BoxBinSize.value()*self.ui.ScrollBar_binchange.value())
    self.current_time_s = self.current_sample//int(self.ui.BoxBinSize.value()) # time in seconds
    self.current_bin = int(self.current_time_s // self.ui.BoxBinSize.value())

    self.ui.ScrollBar_binchange.setValue(self.current_time_s)
    self.ui.lcd_brain_state_previous.display(self.brain_states[self.current_bin - 1])
    self.ui.lcd_brain_state.display(self.brain_states[self.current_bin])
    self.ui.Box_currentTime.setValue(self.ui.ScrollBar_binchange.value())
    self.checkElectrodes()
    self.plot_bin(int(self.ui.ScrollBar_binchange.value()* self.sampling_rate), int((self.ui.ScrollBar_binchange.value() + self.ui.BoxBinSize.value())* self.sampling_rate))
    
  def plot_bin(self, tmin, tmax):
    plt.close('all')
    freq = (self.sampling_rate // 50)*25  # plots different frequency axis depending on the sampling rate.
    #ind_calc[self.currentRecording].plotRawData(self.ui.BoxBinSize.value(), self.ui.ScrollBar_binchange.value(), self.electrodes)
    data_to_plot = self.electrodes_to_plot[:, tmin:tmax]
    times_to_plot = self.times[tmin:tmax]
    
    
    # Plotting all the electrodes (just one bin)
    # just one electrode
    if len(self.electrodes) == 1:
      fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 2))
      ax1.plot(times_to_plot, data_to_plot[0,:])
      ax1.set_xlabel('Time (secs)')
      ax1.set_ylabel(self.elec_names[0])
      ax2.psd(data_to_plot[0,:], Fs = self.sampling_rate)
      ax2.set_ylabel('PSD (dB/Hz)')
      
    
    # more than one electrode
    else:
      fig, axes = plt.subplots(len(self.electrodes), 2, figsize=(10, len(self.electrodes)*2)) #, sharex=True, sharey=True)
      for i in range(len(self.electrodes)):
        axes[i,0].plot(times_to_plot, data_to_plot[i,:])
        axes[i,0].set_xlabel('Time (secs)')
        axes[i,0].set_ylabel(self.elec_names[i])
        axes[i,1].psd(data_to_plot[i,:], Fs = self.sampling_rate)
        axes[i,1].set_ylabel('PSD (dB/Hz)')
        if (i+1) < len(self.electrodes):
          axes[i,0].tick_params(axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                labelbottom=False) # labels along the bottom edge are off
          axes[i,1].tick_params(axis='x', which='both', bottom=False, labelbottom=False) 
    
    plt.show()
    

  def changeRecording(self):
    self.currentRecording = self.ui.ScrollBarCurrentRecord.value()
    self.ui.labelCurrentRecording.setText(self.recordings[self.currentRecording])
    self.current_file = self.file_names[self.currentRecording]
    self.tmax = int(ind_calc[self.currentRecording].rawdata.n_times//self.sampling_rate)
    self.ui.label_Tmax.setText("(" + str(self.tmax) + " s)")
    
    # Reseting scrollbar parameters and setting the maximum according to the selected recording
    self.ui.ScrollBar_binchange.setMaximum(int(ind_calc[self.currentRecording].rawdata.n_times//self.sampling_rate))
    self.ui.ScrollBar_binchange.setValue(0)
    self.ui.ScrollBar_binchange.setSingleStep(self.ui.BoxBinSize.value())
    self.ui.Box_currentTime.setValue(0)
    self.brain_states = np.zeros(int(ind_calc[self.currentRecording].rawdata.n_times//self.sampling_rate + 1))
    self.ui.lcd_brain_state.display(0)
    self.ui.lcd_brain_state_previous.display(0)
    
    # Checking which electrodes are selected, loading just once the data for those electrodes and plotting the first bin
    self.checkElectrodes()
    self.elec_names = ind_calc[self.currentRecording].rawdata.ch_names
    self.times = ind_calc[self.currentRecording].rawdata._times   
    self.plot_bin(0, int(self.ui.BoxBinSize.value()* self.sampling_rate))

  
  
  def addWake(self):
    self.moveforward(0)
  
  def addNoREM(self):
    self.moveforward(1)
  
  def addREM(self):
    self.moveforward(2)
  
  def addOther(self):
    self.moveforward(3)
  
  def addSeizure(self):
    self.moveforward(4)
  
  def moveforward(self, brain_state):
    self.ui.lcd_brain_state_previous.display(brain_state)
    previous_time = self.ui.ScrollBar_binchange.sliderPosition()
    previous_index = int(previous_time//self.ui.BoxBinSize.value())
    self.brain_states[previous_index] = brain_state
    self.ui.lcd_brain_state.display(brain_state)
    self.ui.ScrollBar_binchange.setValue(self.ui.ScrollBar_binchange.value() + self.ui.BoxBinSize.value())
    self.ui.Box_currentTime.setValue(self.ui.ScrollBar_binchange.value())
    self.checkElectrodes()
    tmin = int(self.ui.Box_currentTime.value()*self.sampling_rate)
    tmax = tmin + int(self.ui.BoxBinSize.value()*self.sampling_rate)
    self.plot_bin(tmin, tmax)
    if previous_time + self.ui.BoxBinSize.value() < self.tmax:
      self.ui.lcd_brain_state.display(self.brain_states[previous_index + 1])

  
  def export2excel(self):

    if self.ResultsFolder == '':
      self.ResultsFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    
    df_brain_states = pd.DataFrame({'BrainState': self.brain_states})
    line1 = self.ResultsFolder + self.current_file + '_brain_states_'
    line2 = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + ".xlsx"
    excel_name = line1 + line2
    df_brain_states.to_excel(excel_name, index = False)
    #print(self.brain_states)
    
  def checkElectrodes(self):
    self.electrodes = []
    if self.ui.tabWidget.currentIndex() == 0:  # 32 open ephys
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
    elif self.ui.tabWidget.currentIndex() == 1:  # taini
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
    elif self.ui.tabWidget.currentIndex() == 2: # tetrodes
      if self.ui.tet_1.isChecked(): self.electrodes.append(1)
      if self.ui.tet_2.isChecked(): self.electrodes.append(2)
      if self.ui.tet_3.isChecked(): self.electrodes.append(3)
      if self.ui.tet_4.isChecked(): self.electrodes.append(4)
      if self.ui.tet_5.isChecked(): self.electrodes.append(5)
      if self.ui.tet_6.isChecked(): self.electrodes.append(6)
      if self.ui.tet_7.isChecked(): self.electrodes.append(7)
      if self.ui.tet_8.isChecked(): self.electrodes.append(8)
      if self.ui.tet_9.isChecked(): self.electrodes.append(9)
      if self.ui.tet_10.isChecked(): self.electrodes.append(10)
      if self.ui.tet_11.isChecked(): self.electrodes.append(11)
      if self.ui.tet_12.isChecked(): self.electrodes.append(12)
      if self.ui.tet_13.isChecked(): self.electrodes.append(13)
      if self.ui.tet_14.isChecked(): self.electrodes.append(14)
      if self.ui.tet_15.isChecked(): self.electrodes.append(15)
      if self.ui.tet_16.isChecked(): self.electrodes.append(16)      

      # Changing the electrode voltages to plot each time the electrodes change
      self.electrodes_to_plot = ind_calc[self.currentRecording].rawdata._data[self.electrodes]
      

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
    # Tetrodes
    self.ui.tet_1.setChecked(RightHem)
    self.ui.tet_2.setChecked(RightHem)
    self.ui.tet_3.setChecked(RightHem)
    self.ui.tet_4.setChecked(RightHem)
    self.ui.tet_5.setChecked(RightHem)
    self.ui.tet_6.setChecked(RightHem)
    self.ui.tet_7.setChecked(RightHem)
    self.ui.tet_8.setChecked(RightHem)
    self.ui.tet_9.setChecked(LeftHem)
    self.ui.tet_10.setChecked(LeftHem)
    self.ui.tet_11.setChecked(LeftHem)
    self.ui.tet_12.setChecked(LeftHem)
    self.ui.tet_13.setChecked(LeftHem)
    self.ui.tet_14.setChecked(LeftHem) 
    self.ui.tet_15.setChecked(LeftHem)
    self.ui.tet_16.setChecked(LeftHem)
    

  def closeFigures(self):
    plt.close('all')



ind_calc = [] # list of indiv_tests classes

if __name__=="__main__":
     app = QApplication(sys.argv)
     w = MyForm()
     w.show

     sys.exit(app.exec())
