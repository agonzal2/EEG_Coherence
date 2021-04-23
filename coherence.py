
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

from window_coherence import *

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
from data_classes import session_coherence

montage_name = 'standard_32grid_Alfredo'
#montage_name = 'standard_16grid_taini1'
#montage_name = '/media/jorge/DATADRIVE0/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'

#sampling_rate = 1000
source_name = '100' # how do the name of the recording files begin

class MyForm(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_window_Coherence()
    self.ui.setupUi(self)
    self.ui.ButtonAddExpFolder.clicked.connect(self.addExpFolder)
    self.ui.ButtonDeleteExpFolder.clicked.connect(self.delExpFolder)
    self.ui.ButtonAddControlFolder.clicked.connect(self.addControlFolder)
    self.ui.ButtonDeleteControlFolder.clicked.connect(self.delControlFolder)
    self.ui.pbdestfolder.clicked.connect(self.addResultsFolder)
    self.ui.ButtonRunAll.clicked.connect(self.runAll)
    self.ui.ButtonRunNewBS.clicked.connect(self.runNewBrainState)
    self.ui.ButtonRunNewFreqs.clicked.connect(self.runNewFreqs)
    self.ui.ButtonClearFigures.clicked.connect(self.closeFigures)
    self.ui.Button2PDF.clicked.connect(self.print2pdf)
    self.ui.Button2PNG.clicked.connect(self.print2png)
    # Tab Individual records
    self.ui.ButtonAdd_IndRecords.clicked.connect(self.add_IndRecords)
    self.ui.ButtonDelete_IndRecords.clicked.connect(self.del_IndRecords)
    self.ui.ButtonLoadRecordings.clicked.connect(self.load_recordings)
    self.ui.ButtonPlotRawData.clicked.connect(self.plotRawData)
    self.ui.ButtonPlotPS.clicked.connect(self.plotPS)
    self.ui.radioSelectAll.clicked.connect(self.selectAllElectrodes)
    self.ui.radioSelectNone.clicked.connect(self.selectNoElectrodes)
    self.ui.radioSelectLeftHem.clicked.connect(self.selectLeftHem)
    self.ui.radioSelectRightHem.clicked.connect(self.selectRightHem)
    self.ui.ButtonCloseFigures.clicked.connect(self.closeFigures)
    self.ui.ButtonExportIndPDF.clicked.connect(self.print2pdf)
    self.ui.ButtonExportIndPNG.clicked.connect(self.print2png)
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
    print('loading recordings')
    for i in range(self.ui.listWidget_Indiv_recordings.count()):
      root_dir = str(self.ui.listWidget_Indiv_recordings.item(i).text())
      os.chdir(root_dir)
      recording_folders = next(os.walk('.'))[1] # returns a list of folder in a folder
      for j, folder in enumerate(recording_folders):
        new_recording = indiv_tests(root_dir + "/" + folder, i+j)
        new_recording.load_recordings(montage_name, self.ui.spinBoxDownsampling_2.value())
        ind_calc.append(new_recording)
        self.recordings.append(root_dir + "/" + folder)

    self.changeRecording()
    self.ui.ScrollBarCurrentRecord.setMaximum(len(self.recordings)-1)

  def changeRecording(self):
    self.currentRecording = self.ui.ScrollBarCurrentRecord.value()
    self.ui.labelCurrentRecording.setText(self.recordings[self.currentRecording])
    self.ui.label_Tmax.setText("(" + str(ind_calc[self.currentRecording].raw_data.n_times//1000) + " s)")
    self.ui.BoxTmax.setValue(ind_calc[self.currentRecording].raw_data.n_times//1000)

  def plotRawData(self):
    self.checkElectrodes()
    ind_calc[self.currentRecording].plotRawData(self.ui.BoxBinSize.value(), self.ui.BoxTmin.value(), self.electrodes)

  def plotPS(self):
    self.checkElectrodes()
    ind_calc[self.currentRecording].plotPS(self.ui.BoxTmin.value(), self.ui.BoxTmax.value(), self.electrodes)

  def checkElectrodes(self):
    self.electrodes = []
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

  def selectAllElectrodes(self):
    if self.ui.radioSelectAll.isChecked(): self.changeElectrodeValue(True, True)

  def selectNoElectrodes(self):
    if self.ui.radioSelectNone.isChecked(): self.changeElectrodeValue(False, False)

  def selectLeftHem(self):
    if self.ui.radioSelectLeftHem.isChecked: self.changeElectrodeValue(True, False)

  def selectRightHem(self):
    if self.ui.radioSelectRightHem.isChecked: self.changeElectrodeValue(False, True)

  def changeElectrodeValue(self, LeftHem = True, RightHem = True):
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

  def addExpFolder(self):
    DataFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if DataFolder:
      self.ui.listWidget_Exp.addItem(DataFolder)
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def addResultsFolder(self):
    my_coherence.ResultsFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if my_coherence.ResultsFolder:
      self.ui.listWidget_destfolder.addItem(my_coherence.ResultsFolder)
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def delExpFolder(self):
    self.ui.listWidget_Exp.takeItem(self.ui.listWidget_Exp.currentRow())

  def addControlFolder(self):
    DataFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if DataFolder:
      self.ui.listWidget_Control.addItem(DataFolder)
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def delControlFolder(self):
    self.ui.listWidget_Control.takeItem(self.ui.listWidget_Control.currentRow())

  def plotShortDistance(self):
    my_coherence.plot_mean_short_distance()

  def plotLongDistance(self):
    my_coherence.plot_mean_long_distance()

  def runAll(self):

    my_coherence.srate = self.ui.spinBoxSamplingRate.value()
    my_coherence.downsamp = self.ui.spinBoxDownsampling.value()
    my_coherence.final_srate = my_coherence.srate/my_coherence.downsamp

    # Goes through all the excel files with the brain states
    # If an animal has more than one recording, it store them together
    # Then it goes through all the folders for those rats with excel files
    # with brain states
    # dividing files by animal (KO)
    for i in range(self.ui.listWidget_Exp.count()):
      os.chdir(str(self.ui.listWidget_Exp.item(i).text()))
      d = os.getcwd() + "/" # "\\" for Windows
      matching_files = glob.glob(r'*npy')
      for file_m in matching_files:
        if file_m[0:5] not in my_coherence.l_prefixes_KO:
          my_coherence.l_prefixes_KO.append(file_m[0:5])

      # gathering the recordings per animal (they will be analysed together)
      for pattern in my_coherence.l_prefixes_KO:
        matching_files = glob.glob(r'*' + pattern + r'*')
        l_animal_npy_files = []
        for matching_file in matching_files:
            if 'npy' in matching_file:
                l_animal_npy_files.append(d+matching_file)            

        my_coherence.l_npy_files_KO.append(l_animal_npy_files)

    # dividing files by animal (WT)
    for i in range(self.ui.listWidget_Control.count()):
      os.chdir(str(self.ui.listWidget_Control.item(i).text()))
      d = os.getcwd() + "/" # "\\"
      matching_files = glob.glob(r'*npy')
      for file_m in matching_files:
        if file_m[0:5] not in my_coherence.l_prefixes_WT:
          my_coherence.l_prefixes_WT.append(file_m[0:5])

      # gathering the recordings per animal (they will be analysed together)
      for pattern in my_coherence.l_prefixes_WT:
        matching_files = glob.glob(r'*' + pattern + r'*')
        l_animal_npy_files = []
        for matching_file in matching_files:
            if 'npy' in matching_file:
                l_animal_npy_files.append(d+matching_file)
            
        my_coherence.l_npy_files_WT.append(l_animal_npy_files)

    if self.ui.radioButtonOpenEphys.isChecked():
      my_coherence.montage_name = '/media/jorge/DATADRIVE0/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'
      my_coherence.recording_type = 'openephys'
      my_coherence.n_electrodes = 32
    elif self.ui.radioButtonTaini.isChecked():
      my_coherence.montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
      my_coherence.recording_type = 'taini'
      my_coherence.n_electrodes = 14

    # First it is called the downsampling
    my_coherence.loadnpydatatomne()

    #Once it finished the load of data, allow to repeat analysis without loading it again.
    self.ui.ButtonRunNewBS.setEnabled(True)
    self.ui.ButtonRunNewFreqs.setEnabled(True)
    #self.ui.ButtonRunAll.setDisabled(True)
    self.set_brain_state()
    # Esploratory analysis
    if self.ui.checkBoxExploratoryAnalysis.isChecked():
      my_coherence.coh_type = 'imag'
      for longD in np.arange(2, 5.5, 1):
        self.ui.SpinBoxLongDist.setValue(longD)
        self.runNewBrainState()
        file_name = my_coherence.coh_type + '_Dist' + str(longD) + '_' + self.brain_state_name
        self.print2pdf(file_name)
        self.closeFigures()
    # Individual analysis
    else:      
      if self.ui.rbcohabs.isChecked():
        my_coherence.coh_type = 'abs'
      else:
        my_coherence.coh_type = 'imag'
      self.runNewBrainState()


  def set_brain_state(self):
    # brain state can take autoexclusive values from 0 to 5
    self.brain_state = 0*True + 1*self.ui.radioButtonNoREM.isChecked() + 2*self.ui.radioButtonREM.isChecked() \
                  + 3*self.ui.radioButtonSleeping.isChecked() + 4*self.ui.radioButtonConvulsion.isChecked() \
                  + 5*self.ui.radioButtonNonConvulsive.isChecked()

    self.brain_state_name = self.ui.radioButtonWake.text()*self.ui.radioButtonWake.isChecked() \
                      + self.ui.radioButtonNoREM.text()*self.ui.radioButtonNoREM.isChecked() \
                      + self.ui.radioButtonREM.text()*self.ui.radioButtonREM.isChecked() \
                      + self.ui.radioButtonSleeping.text()*self.ui.radioButtonSleeping.isChecked() \
                      + self.ui.radioButtonConvulsion.text()*self.ui.radioButtonConvulsion.isChecked() \
                      + self.ui.radioButtonNonConvulsive.text()*self.ui.radioButtonNonConvulsive.isChecked()

  def runNewBrainState(self):
    # calculate combinations depending on the short or long distance criteria
    my_coherence.calc_combinations(self.ui.SpinBoxNeighborDist.value(), self.ui.SpinBoxLongDist.value())

    # calculate coefficients of notch filter at 50 Hz
    my_coherence.calc_notch(self.ui.spinBoxNotchQ.value(), self.ui.spinBoxDownsampling.value())

    frequency_list = []
    frequency_list = self.get_frequency_bands()

    my_coherence.calc_z_coh(frequency_list, self.brain_state_name, self.brain_state, self.ui.spinLongProcesses.value(), self.ui.spinLongChunksize.value(),
                            self.ui.spinShortProcesses.value(), self.ui.spinShortChunksize.value())

    self.freq_list_results = my_coherence.return_freq_results()
    self.write_table_results()


  def runNewFreqs(self):
    frequency_list = []
    frequency_list = self.get_frequency_bands()

    my_coherence.calc_zcoh_freq_bands(frequency_list)

    self.freq_list_results = my_coherence.return_freq_results()
    self.write_table_results()


  def closeFigures(self):
    plt.close('all')

  def print2pdf(self, filename=""):

    if filename:
      filename2 = '/' + filename + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.pdf'
      pdf = matplotlib.backends.backend_pdf.PdfPages(my_coherence.ResultsFolder + filename2)
      figs = [plt.figure(n) for n in plt.get_fignums()]
      for fig in figs:
        fig.savefig(pdf, format='pdf')
      pdf.close()
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def print2png(self):
    my_coherence.figFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    if my_coherence.figFolder:
      prefix = '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
      for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(my_coherence.figFolder + prefix +'figure%d.png' % i)
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def get_frequency_bands(self):
    nrows = self.ui.tableFrequencies.rowCount()
    freq_list = [] # list of 6 element tuples, frequency bands and the results for those bands

    for row in range(1, nrows):
      band_name = self.ui.tableFrequencies.item(row, 0).text()

      if band_name == "":
        print (row - 1), 'frequency bands'
        break

      band_freq_from = self.ui.tableFrequencies.item(row, 1).text()
      band_freq_to = self.ui.tableFrequencies.item(row, 2).text()
      freq_list.append((band_name, int(band_freq_from), int(band_freq_to))) # double (()) is necessary

    return freq_list

  def write_table_results(self):

    for n, freq_interval_results in enumerate(self.freq_list_results):
      self.ui.tableFrequencies.setItem(n+1, 3, QTableWidgetItem(str(freq_interval_results[0])))
      self.ui.tableFrequencies.setItem(n+1, 4, QTableWidgetItem(str(freq_interval_results[1])))
      self.ui.tableFrequencies.setItem(n+1, 5, QTableWidgetItem(str(freq_interval_results[2])))


class coherence_eeg ():
  folders_data_KO = []
  folders_data_WT = []
  xlsfiles_KO = []
  xlsfiles_WT = []
  all_KO_coh_data = [] # list that would contain all the instances for the KO sessions analysed data
  all_WT_coh_data = [] # list for WT
  short_d_comb = [] # all the short distance combinations between electrodes
  long_d_comb = []
  freq_array = [] #  to plot the frequency axis
  brain_state_name = ""
  f_ratio = 1
  coh_type = 'abs'
  ResultsFolder = ""
  b = np.array([0,0,0]) # notch filter parameter
  a = np.array([0,0,0]) # notch filter parameter
  l_prefixes_KO = []
  l_npy_files_KO = []
  l_prefixes_WT = []
  l_npy_files_WT = []
  second_parts_cntrl = [] # to join parts A and B of split recordings
  second_parts_exp = []
  n_electrodes = 32
  n_aux_elec = 3
  recording_type = ''
  srate = 1000
  downsamp = 1
  final_srate = 1000


  def __init__(self, brain_state = 0, montage_name = 'standard_32grid_Alfredo'):
    self.brain_state = brain_state
    self.montage_name = montage_name

  
  def get_join_data_KO(self, i, animal):

    print(f"loading recordings for the KO animal: {self.l_prefixes_KO[i]}")
    l_raw_data = []
    l_raw_times = []
    l_raw_data = []
    l_electrode_data = [list() for _ in range(self.n_electrodes)]
    for npy_file in animal:
      # first we join all the animal times
      if self.recording_type == 'openephys':
        raw_list = npy32mne(npy_file, self.montage_name)
      elif self.recording_type == 'taini':
        raw_list = taininumpy2mne(npy_file, self.montage_name, self.final_srate)

      l_raw_times = l_raw_times + np.ndarray.tolist(raw_list._times)
      # and now the data, electrode by electrode
      # joining all the data across all the electrode recordings belonging to an animal
      for elect in range(self.n_electrodes):
          l_electrode_data[elect] = l_electrode_data[elect] + np.ndarray.tolist(raw_list._data[elect,:])
    # list of of the data joined electrodes of an animal
    for elect in range(self.n_electrodes):
      l_raw_data.append(l_electrode_data[elect])
    join_raw_times = np.asarray(l_raw_times)
    join_raw_data = np.asarray(l_raw_data)
    Cxy = session_coherence(join_raw_times, join_raw_data, self.downsamp,
                              self.montage_name, self.n_electrodes, self.srate, self.brain_state)
    
    self.all_KO_coh_data.append(Cxy) # Appending the class instance for every KO session
    del raw_list # saving memory
    del Cxy
    del join_raw_data
    del join_raw_times


  def get_join_data_WT(self, i, animal):
    print(f"loading recordings for the WT animal: {self.l_prefixes_WT[i]}")
    l_raw_data = []
    l_raw_times = []
    l_raw_data = []
    l_electrode_data = [list() for _ in range(self.n_electrodes)]
    for npy_file in animal:
      # first we join all the animal times
      if self.recording_type == 'openephys':
        raw_list = npy32mne(npy_file, self.montage_name)
      elif self.recording_type == 'taini':
        raw_list = taininumpy2mne(npy_file, self.montage_name, self.final_srate)
      
      l_raw_times = l_raw_times + np.ndarray.tolist(raw_list._times)
      # and now the data, electrode by electrode
      # joining all the data across all the electrode recordings belonging to an animal
      for elect in range(self.n_electrodes):
          l_electrode_data[elect] = l_electrode_data[elect] + np.ndarray.tolist(raw_list._data[elect,:])
    # list of of the data joined electrodes of an animal
    for elect in range(self.n_electrodes):
      l_raw_data.append(l_electrode_data[elect])
    join_raw_times = np.asarray(l_raw_times)
    join_raw_data = np.asarray(l_raw_data)
    Cxy = session_coherence(join_raw_times, join_raw_data, self.downsamp,
                              self.montage_name, self.n_electrodes, self.srate, self.brain_state)
    
    self.all_WT_coh_data.append(Cxy) # Appending the class instance for every WT session
    del raw_list # saving memory
    del Cxy
    del join_raw_data
    del join_raw_times


  # loads the npy data that should be downsampled and split by brain state already
  def loadnpydatatomne(self):
    
    # Creating instances to analyse all the recordings per animal
    start_timeKO = time.time()
    for i, animal in enumerate(self.l_npy_files_KO):
      self.get_join_data_KO(i, animal)

    print(f'KO sessions loaded and converted to mne format {(time.time() - start_timeKO)} seconds ---')

    start_timeWT = time.time()
    for i, animal in enumerate(self.l_npy_files_WT):
      self.get_join_data_WT(i, animal)

    print(f'WT sessions loaded and converted to mne format {(time.time() - start_timeWT)} seconds ---')
    print(f'Total loading time: {(time.time() - start_timeKO)} seconds ---')

  # Now methods that modify the analysis once all the data is loaded and downsampled.
  # So the notch filter, the distance range or the brain state can be modified without loading
  # again all the data
  def calc_notch(self, Q, downsampling):
    self.b, self.a = iirnotch(50.0, Q, self.final_srate)

  # calculating the different combinations between electrodes, short and long distance
  def calc_combinations(self, neighbors_dist, long_distance):
    self.long_distance = long_distance
    self.short_d_comb, self.long_d_comb = electrode_combinations(self.montage_name, neighbors_dist, long_distance, self.recording_type, self.n_electrodes)

  # It gives the chance to change the brain state every time it is called.
  def calc_z_coh(self, f_l, brain_state_name, brain_state = 0, l_processes = 48, l_chunk = 24, s_processes = 12, s_chunk = 12):
    self.brain_state = brain_state
    self.brain_state_name = brain_state_name
    self.freq_list = f_l
    for n, Cxy in enumerate(self.all_KO_coh_data):
      Cxy.calc_cohe_long(self.long_d_comb, l_processes, l_chunk, self.b, self.a, self.coh_type)
      Cxy.calc_cohe_short(self.short_d_comb, s_processes, s_chunk, self.b, self.a, self.coh_type)

    for n, Cxy in enumerate(self.all_WT_coh_data):
      Cxy.calc_cohe_long(self.long_d_comb, l_processes, l_chunk, self.b, self.a, self.coh_type)
      Cxy.calc_cohe_short(self.short_d_comb, s_processes, s_chunk, self.b, self.a, self.coh_type)

    self.calc_zcoh_freq_bands(f_l)


  # Once the coherence is calculated, we split the spectrum in frequency bands.
  def calc_zcoh_freq_bands(self, f_l):
    self.freq_list = f_l
    for Cxy in self.all_KO_coh_data:
      Cxy.calc_zcoh_long(self.freq_list)
      Cxy.calc_zcoh_short(self.freq_list)

    for Cxy in self.all_WT_coh_data:
      Cxy.calc_zcoh_long(self.freq_list)
      Cxy.calc_zcoh_short(self.freq_list)
      if len(Cxy.f_w > 0) : self.f_array = Cxy.f_w

    self.f_ratio = Cxy.f_ratio
    self.calc_mean_coh()


  def calc_mean_coh(self):
    # Two ways Anova Dataframe
    cols = ('band_mean', 'genotype', 'freq_band')
    anova_lst_short = []
    anova_lst_long = []
    anova_lst_ratio = []

    # Once all the sessions coherence z-transfroms are calculated, the mean calculated in z, and
    # transformed back. We can plot the results and do statistics with the differences between samples
    KO_all_short_lines_m = []
    KO_all_long_lines_m = []
    KO_times = []
    self.mean_KO_sho_plot_line = []
    self.sem_KO_sho_plot_line = []
    self.mean_KO_lon_plot_line = []
    self.sem_KO_lon_plot_line = []
    self.mean_WT_sho_plot_line = []
    self.sem_WT_sho_plot_line = []
    self.mean_WT_lon_plot_line = []
    self.sem_WT_lon_plot_line = []

    # n should be the amount of recordings. So here we have grouped by WT (short and long range)
    # and KO (short and long range). First KO
    KO_all_short_bands_m_t = []
    KO_all_long_bands_m_t = []
    for n, Cxy in enumerate(self.all_KO_coh_data):
      if Cxy.time_state == 0: continue # sometimes one of the brain state does not occur during a recording
      KO_times.append(Cxy.time_state) # lists of weights to do weighted mean and sem
      KO_all_short_lines_m.append(Cxy.short_line_plot_1rec_m)
      KO_all_long_lines_m.append(Cxy.long_line_plot_1rec_m)
      # for plotting short-range
      KO_short_bands_m = []
      KO_long_bands_m = []
      for n, freq_band in enumerate(self.freq_list):
        KO_short_bands_m.append(Cxy.short_1rec_m[n])
        KO_long_bands_m.append(Cxy.long_1rec_m[n])
        if (n>0):
          anova_lst_short.append([Cxy.time_state*Cxy.short_1rec_m[n], freq_band[0], 'KO'])
          anova_lst_long.append([Cxy.time_state*Cxy.long_1rec_m[n], freq_band[0], 'KO'])
          anova_lst_ratio.append([(Cxy.long_1rec_m[n]/Cxy.short_1rec_m[n]), freq_band[0], 'KO'])

      KO_all_short_bands_m_t.append(KO_short_bands_m)
      KO_all_long_bands_m_t.append(KO_long_bands_m)

    # Lists of lists need to be transposed so in each list there are the values of a single freq band
    KO_all_short_bands_m = np.array(KO_all_short_bands_m_t).T.tolist()
    KO_all_long_bands_m = np.array(KO_all_long_bands_m_t).T.tolist()

    total_time = np.sum(np.array(KO_times))
    KO_weights = (n+1)*np.array(KO_times)/total_time

    # Statistics with all the recordings (mean and standard error of the mean)
    # Line plots (weighted)
    for row in np.array(KO_all_short_lines_m).T:
      row_mean, row_sem = weighted_avg_sem(row, KO_weights, "KOshortlines")
      self.mean_KO_sho_plot_line.append(row_mean)
      self.sem_KO_sho_plot_line.append(row_sem)

    for row in np.array(KO_all_long_lines_m).T:
      row_mean, row_sem = weighted_avg_sem(row, KO_weights, "KOlonglines")
      self.mean_KO_lon_plot_line.append(row_mean)
      self.sem_KO_lon_plot_line.append(row_sem)

    # Bar plots and significance tests (a single final value per freq band)
    self.mean_KO_sho = []
    self.sem_KO_sho = []
    self.mean_KO_lon = []
    self.sem_KO_lon = []
    self.mean_ratio_KO = []
    self.sem_ratio_KO = []
    for n, freq_band in enumerate(self.freq_list):
      mean_KO_sho, sem_KO_sho = weighted_avg_sem(KO_all_short_bands_m[n], KO_weights, "so")
      self.mean_KO_sho.append(mean_KO_sho)
      self.sem_KO_sho.append(sem_KO_sho)
      mean_KO_lon, sem_KO_lon = weighted_avg_sem(KO_all_long_bands_m[n], KO_weights, "lo")
      self.mean_KO_lon.append(mean_KO_lon)
      self.sem_KO_lon.append(sem_KO_lon)
      mean_KO_ratio_lon_sho, sem_KO_ratio_lon_sho = weighted_avg_sem(
                  np.divide(KO_all_long_bands_m[n], KO_all_short_bands_m[n]), KO_weights, "ratio")
      self.mean_ratio_KO.append(mean_KO_ratio_lon_sho)
      self.sem_ratio_KO.append(sem_KO_ratio_lon_sho)

    # Same for WT
    WT_all_short_lines_m = []
    WT_all_long_lines_m = []
    WT_times = []
    self.mean_WT_sho_plot_line = []
    self.sem_WT_sho_plot_line = []
    self.mean_WT_lon_plot_line = []
    self.sem_WT_lon_plot_line = []
    self.mean_WT_sho_plot_line = []
    self.sem_WT_sho_plot_line = []
    self.mean_WT_lon_plot_line = []
    self.sem_WT_lon_plot_line = []

    WT_all_short_bands_m_t = []
    WT_all_long_bands_m_t = []
    for n, Cxy in enumerate(self.all_WT_coh_data):
      if Cxy.time_state == 0: continue
      WT_times.append(Cxy.time_state) # lists of weights to do weighted mean and sem
      WT_all_short_lines_m.append(Cxy.short_line_plot_1rec_m)
      WT_all_long_lines_m.append(Cxy.long_line_plot_1rec_m)
      # for plotting short-range
      WT_short_bands_m = []
      WT_long_bands_m = []
      for n, freq_band in enumerate(self.freq_list):
        WT_short_bands_m.append(Cxy.short_1rec_m[n])
        WT_long_bands_m.append(Cxy.long_1rec_m[n])
        if (n>0):
          anova_lst_short.append([Cxy.time_state*Cxy.short_1rec_m[n], freq_band[0], 'WT'])
          anova_lst_long.append([Cxy.time_state*Cxy.long_1rec_m[n], freq_band[0], 'WT'])
          anova_lst_ratio.append([(Cxy.long_1rec_m[n]/Cxy.short_1rec_m[n]), freq_band[0], 'WT'])

      WT_all_short_bands_m_t.append(WT_short_bands_m)
      WT_all_long_bands_m_t.append(WT_long_bands_m)

    WT_all_short_bands_m = np.array(WT_all_short_bands_m_t).T.tolist()
    WT_all_long_bands_m = np.array(WT_all_long_bands_m_t).T.tolist()

    total_time = np.sum(np.array(WT_times))
    WT_weights = (n+1)*np.array(WT_times)/total_time

    # Statistics with all the recordings (mean and standard error of the mean)
    # Line plots (weighted)
    for row in np.array(WT_all_short_lines_m).T:
      row_mean, row_sem = weighted_avg_sem(row, WT_weights, "WTshortlines")
      self.mean_WT_sho_plot_line.append(row_mean)
      self.sem_WT_sho_plot_line.append(row_sem)

    for row in np.array(WT_all_long_lines_m).T:
      row_mean, row_sem = weighted_avg_sem(row, WT_weights, "WTlonglines")
      self.mean_WT_lon_plot_line.append(row_mean)
      self.sem_WT_lon_plot_line.append(row_sem)

    # Bar plots and significance tests (a single final value per freq band)
    self.mean_WT_sho = []
    self.sem_WT_sho = []
    self.mean_WT_lon = []
    self.sem_WT_lon = []
    self.mean_ratio_WT = []
    self.sem_ratio_WT = []
    for n, freq_band in enumerate(self.freq_list):
      mean_WT_sho, sem_WT_sho = weighted_avg_sem(WT_all_short_bands_m[n], WT_weights, "so")
      self.mean_WT_sho.append(mean_WT_sho)
      self.sem_WT_sho.append(sem_WT_sho)
      mean_WT_lon, sem_WT_lon = weighted_avg_sem(WT_all_long_bands_m[n], WT_weights, "lo")
      self.mean_WT_lon.append(mean_WT_lon)
      self.sem_WT_lon.append(sem_WT_lon)
      mean_WT_ratio_lon_sho, sem_WT_ratio_lon_sho = weighted_avg_sem(
                  np.divide(WT_all_long_bands_m[n], WT_all_short_bands_m[n]), WT_weights, "ratio")
      self.mean_ratio_WT.append(mean_WT_ratio_lon_sho)
      self.sem_ratio_WT.append(sem_WT_ratio_lon_sho)

    ######################
    # Anova calculations #
    # short-range
    self.df_short = pd.DataFrame(anova_lst_short, columns=cols)
    print('')
    print('###### Dataframe for Short-range ######')
    print(self.df_short)
    formula = 'band_mean~C(genotype)+C(freq_band)+C(genotype):C(freq_band)'
    model_short = ols(formula, self.df_short).fit()
    aov_table_short = anova_lm(model_short, typ=2)
    print(aov_table_short)
    # long range
    self.df_long = pd.DataFrame(anova_lst_long, columns=cols)
    print('')
    print('###### Dataframe for Long-range ######')
    print(self.df_long)
    model_long = ols(formula, self.df_long).fit()
    aov_table_long = anova_lm(model_long, typ=2)
    print(aov_table_long)
    # ratio long/short
    self.df_ratio = pd.DataFrame(anova_lst_ratio, columns=cols)
    print('')
    print('###### Dataframe for Ratio Long/Short ######')
    print(self.df_ratio)
    model_ratio = ols(formula, self.df_ratio).fit()
    aov_table_ratio = anova_lm(model_ratio, typ=2)
    print(aov_table_ratio)

    print('')
    print('##########################')
    print('T-tests per frequency band')
    print('##########################')
    # Short range (weighted)
    KO_w = KO_weights
    WT_w = WT_weights
    # Loop over every frequency band, short, long and range ratio
    self.list_freq_results = []
    for n, freq_band in enumerate(self.freq_list):
      band_results = []
      stat, p, dgf = wttest(np.array(KO_all_short_bands_m[n]),
                                np.array(WT_all_short_bands_m[n]), alternative='two-sided',
                                usevar='pooled', weights=(KO_w, WT_w), value=0)
      print('T-test %s Short-Range = %.3f, p = %.3f, dgf=%.3f' % (freq_band[0],stat, p, dgf))
      band_results.append(p)
      stat, p, dgf = wttest(np.array(KO_all_long_bands_m[n]),
                                np.array(WT_all_long_bands_m[n]), alternative='two-sided',
                                usevar='pooled', weights=(KO_w, WT_w), value=0)
      print('T-test %s Long-Range = %.3f, p = %.3f, dgf=%.3f' % (freq_band[0],stat, p, dgf))
      band_results.append(p)
      stat, p, dgf = wttest(np.array(np.divide(KO_all_long_bands_m[n], KO_all_short_bands_m[n])),
                              np.array(np.divide(WT_all_long_bands_m[n], WT_all_short_bands_m[n])),
                              alternative='two-sided', usevar='pooled', weights=(KO_w, WT_w), value=0)
      print('T-test %s Ratio Long/Short range = %.3f, p = %.3f, dgf=%.3f' % (freq_band[0],stat, p, dgf))
      band_results.append(p)
      self.list_freq_results.append(band_results)

    self.plot_mean_short_distance()
    self.plot_mean_long_distance()
    self.plot_bars()
    self.bar_plot_ratio_long_short()

    # Plotting individual coherences
    self.plot_individual_zCoh_short_WT()
    self.plot_individual_zCoh_long_KO()
    self.plot_individual_zCoh_short_KO()
    self.plot_individual_zCoh_long_WT()

    # Exporting individual coherences to excel
    line1 = self.ResultsFolder + '/z_Individual_Coh_' + self.brain_state_name + "_" + str(self.long_distance) + "_"
    line2 = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + "_" + self.coh_type + ".xlsx"
    excel_name = line1 + line2
    df_to_excel(excel_name, self.df_shortwt, "ShortWT")
    df_to_excel(excel_name, self.df_longwt, "LongWT")
    df_to_excel(excel_name, self.df_shortko, "ShortKO")
    df_to_excel(excel_name, self.df_longko, "LongKO")

    #self.plot_WT_average_KO_indiv_short()
    self.plot_WT_average_KO_indiv_long()
    #self.plot_corr_convulsions_coh()

    # Plotting Power Spectrum of the first channel of every rat recording (per brain state)
    self.plot_power_spectrums()

  def return_freq_results(self):
    return self.list_freq_results

  def plot_power_spectrums(self):
    ax = self.indiv_fig()
    for Cxy in self.all_WT_coh_data:
      ax.psd(Cxy.volt_state[:,0], 256, Cxy.final_srate)

    plt.title('PSD %s WT' %self.brain_state_name, fontsize=18)
    ax.set_xlim([0,Cxy.set_top_freq()])
    ax.legend()
    plt.show()

    bx = self.indiv_fig()
    for Cxy in self.all_KO_coh_data:
      bx.psd(Cxy.volt_state[:,0], 256, Cxy.final_srate)

    plt.title('PSD %s KO' %self.brain_state_name, fontsize=18)
    bx.set_xlim([0,Cxy.set_top_freq()])
    bx.legend()
    plt.show()

  def plot_mean_short_distance(self):
    f = self.f_array
    plt.figure(figsize=(20,10))
    plt.plot(f, self.mean_KO_sho_plot_line, label = 'short-range KO', color='red')
    plt.plot(f, self.mean_WT_sho_plot_line, label = 'short-range WT', color='black')
    plt.fill_between(f, np.array(self.mean_KO_sho_plot_line) - np.array(self.sem_KO_sho_plot_line),
                        np.array(self.mean_KO_sho_plot_line) + np.array(self.sem_KO_sho_plot_line),
                        edgecolor='#efb7b2', facecolor= '#efb7b2',
                        linewidth=1, linestyle='dashed',antialiased=False)
    plt.fill_between(f, np.array(self.mean_WT_sho_plot_line) - np.array(self.sem_WT_sho_plot_line),
                        np.array(self.mean_WT_sho_plot_line) + np.array(self.sem_WT_sho_plot_line),
                        edgecolor='#bbbbbb', facecolor= '#bbbbbb',
                        linewidth=1, antialiased=False) # linestyle='dashed',

    plt.title('Coherence Short-range %s' %self.brain_state_name, fontsize=18)
    max_value = max(max(self.mean_KO_sho_plot_line), max(self.mean_WT_sho_plot_line)) \
                + max(max(self.sem_KO_sho_plot_line), max(self.sem_WT_sho_plot_line))
    self.rest_of_the_plot(max_value)

  def plot_mean_long_distance(self):
    f = self.f_array
    plt.figure(figsize=(20,10))
    plt.plot(f, self.mean_KO_lon_plot_line, label = 'long-range KO', color='red')
    plt.plot(f, self.mean_WT_lon_plot_line, label = 'long-range WT', color='black')
    plt.fill_between(f, np.array(self.mean_KO_lon_plot_line) - np.array(self.sem_KO_lon_plot_line),
                        np.array(self.mean_KO_lon_plot_line) + np.array(self.sem_KO_lon_plot_line),
                        edgecolor='#efb7b2', facecolor= '#efb7b2',
                        linewidth=1, linestyle='dashed', antialiased=False)
    plt.fill_between(f, np.array(self.mean_WT_lon_plot_line) - np.array(self.sem_WT_lon_plot_line),
                        np.array(self.mean_WT_lon_plot_line) + np.array(self.sem_WT_lon_plot_line),
                        edgecolor='#bbbbbb', facecolor= '#bbbbbb',
                        linewidth=1, linestyle='dashed', antialiased=False)


    plt.title('Coherence Long-range %s' %self.brain_state_name, fontsize=18)
    max_value = max(max(self.mean_KO_lon_plot_line), max(self.mean_WT_lon_plot_line)) \
                + max(max(self.sem_KO_lon_plot_line), max(self.sem_WT_lon_plot_line))
    self.rest_of_the_plot(max_value)

  def rest_of_the_plot(self, max_value):
    if self.coh_type == 'abs':
      max_z1 = 1 #np.max(Cxy_mean_short_rem_13)
    else:
      max_z1 = max_value
    min_z1 = 0 #np.min(Cxy_mean_short_rem_13)

    #labels = []
    for n, freq_band in enumerate(self.freq_list):
      if n>0:
        position = int(freq_band[1]+ 0.4*(freq_band[2]-freq_band[1]))
        plt.text(position, min_z1, freq_band[0], family='arial', fontsize=18)
      #labels.append(freq_band[0]) #'Overall',  'low freqs', 'low ' r'$\gamma$', 'high ' r'$\gamma$'] # r'$\delta$', r'$\theta$', r'$\sigma$', r'$\beta$
      plt.plot([freq_band[1], freq_band[1]], [min_z1, max_z1], color = '#888888', linestyle='dashed', lw=1)
      plt.plot([freq_band[2], freq_band[2]], [min_z1, max_z1], color = '#888888', linestyle='dashed', lw=1)

    plt.legend(fontsize=18)
    plt.xlabel('frequency [Hz]', fontsize=18)
    plt.ylabel('Average $z^{-1}$ Coherence', fontsize=18)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.tick_params(axis = 'both', which = 'minor', labelsize = 16)

    plt.show()

  def plot_bars(self):

    labels = []
    for freq_band in self.freq_list:
      labels.append(freq_band[0]) #'Overall',  'low freqs', 'low ' r'$\gamma$', 'high ' r'$\gamma$'] # r'$\delta$', r'$\theta$', r'$\sigma$', r'$\beta$

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, self.mean_KO_sho, width, yerr=self.sem_KO_sho,  capsize = 3, label='KO', color='red')
    rects2 = ax.bar(x + width/2, self.mean_WT_sho, width, yerr=self.sem_WT_sho,  capsize = 3, label='WT', color='black')

    ## Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean $z^{-1}$ Coherence')
    ax.set_title('Short-range coherence %s' %self.brain_state_name, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plt.show()

    fig2, ax2 = plt.subplots()
    rects1 = ax2.bar(x - width/2, self.mean_KO_lon, width, yerr=self.sem_KO_lon,  capsize = 3, label='KO', color='red')
    rects2 = ax2.bar(x + width/2, self.mean_WT_lon, width, yerr=self.sem_WT_lon,  capsize = 3, label='WT', color='black')

    ## Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('Mean $z^{-1}$ Coherence')
    ax2.set_title('Long-range coherence %s' %self.brain_state_name, fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    fig2.tight_layout()

    plt.show()


  def bar_plot_ratio_long_short(self):

    labels = []
    for freq_band in self.freq_list:
      labels.append(freq_band[0]) #'Overall',  'low freqs', 'low ' r'$\gamma$', 'high ' r'$\gamma$'] # r'$\delta$', r'$\theta$', r'$\sigma$', r'$\beta$

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, self.mean_ratio_KO, width, yerr=self.sem_ratio_KO,  capsize = 3, label='KO', color='red')
    rects2 = ax.bar(x + width/2, self.mean_ratio_WT, width, yerr=self.sem_ratio_WT,  capsize = 3, label='WT', color='black')

    ## Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Ratio')
    ax.set_title('Long-range/short-range ratio %s' %self.brain_state_name, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    plt.show()

  #def new_excel_sheet(self, sh_name):
  #  worksheet_ti= self.workbook.add_worksheet(sh_name)
  #  worksheet_ti.write(0, 0, 'Freq (Hz)')
  #  for n, freq in enumerate(self.f_array):
  #    worksheet_ti.write(n+1, 0, freq)
  #
  #  return worksheet_ti

  def plot_individual_zCoh_short_WT(self):

    ax = self.indiv_fig()
    #worksheet_ti = self.new_excel_sheet('shortWt')
    data_f = pd.DataFrame({'Freqs': self.f_array})

    for n, Cxy in enumerate(self.all_WT_coh_data):
      if len(Cxy.short_line_plot_1rec_m) > 0:
        ax.plot(self.f_array, Cxy.short_line_plot_1rec_m, label = str(n), linewidth=0.5)
        #worksheet_ti.write(0, n+1, 'n: ' + str(n+1))
        data_f['n' + str(n+1)]= Cxy.short_line_plot_1rec_m
        #worksheet_ti.write(i+1, n+1, cohe)

    title_plot = 'Short-range WT ' + self.brain_state_name
    self.indiv_plots_common(ax, title_plot)
    print('Individual Coherences ShortWT')
    print(data_f.to_string())
    self.df_shortwt = data_f
    print('')



  def plot_individual_zCoh_short_KO(self):

    ax = self.indiv_fig()
    data_f = pd.DataFrame({'Freqs': self.f_array})

    for n, Cxy in enumerate(self.all_KO_coh_data):
      if len(Cxy.short_line_plot_1rec_m) > 0:
        ax.plot(self.f_array, Cxy.short_line_plot_1rec_m, label = str(n), linewidth=0.5)
        data_f['n' + str(n+1)]= Cxy.short_line_plot_1rec_m

    title_plot = 'Short-range KO ' + self.brain_state_name
    self.indiv_plots_common(ax, title_plot)
    print('Individual Coherences ShortKO')
    print(data_f.to_string())
    self.df_shortko = data_f
    print('')



  def plot_individual_zCoh_long_KO(self):

    ax = self.indiv_fig()
    data_f = pd.DataFrame({'Freqs': self.f_array})

    for n, Cxy in enumerate(self.all_KO_coh_data):
      if len(Cxy.long_line_plot_1rec_m) > 0:
        ax.plot(self.f_array, Cxy.long_line_plot_1rec_m, label = str(n), linewidth=0.5)
        data_f['n' + str(n+1)]= Cxy.long_line_plot_1rec_m

    title_plot = 'Long-range KO ' + self.brain_state_name
    self.indiv_plots_common(ax, title_plot)
    print('Individual Coherences LongKO')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      print(data_f)
    print('')
    self.df_longko = data_f



  def plot_individual_zCoh_long_WT(self):

    ax = self.indiv_fig()
    data_f = pd.DataFrame({'Freqs': self.f_array})

    for n, Cxy in enumerate(self.all_WT_coh_data):
      if len(Cxy.long_line_plot_1rec_m) > 0:
        plt.plot(self.f_array, Cxy.long_line_plot_1rec_m, label = str(n), linewidth=0.5)
        data_f['n' + str(n+1)]= Cxy.long_line_plot_1rec_m

    title_plot = 'Long-range WT ' + self.brain_state_name
    self.indiv_plots_common(ax, title_plot)
    print('Individual Coherences Long WT')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
      print(data_f)
    print('')
    self.df_longwt = data_f



  #def plot_WT_average_KO_indiv_short(self):

  def plot_corr_convulsions_coh(self):
    ax = self.indiv_fig()
    coh_l = []
    times_l = []
    for n, Cxy in enumerate(self.all_KO_coh_data):
      if len(Cxy.long_line_plot_1rec_m) > 0:
        coherences = np.average(Cxy.long_line_plot_1rec_m)
        convulsion_time = 100*(Cxy.time_convulsion/(Cxy.time_convulsion + Cxy.time_non_convulsion))
        coh_l.append(coherences)
        times_l.append(convulsion_time)
        ax.plot(convulsion_time, coherences, 'o', label = str(n))

    max_time = max(times_l)*1.1
    max_coh = max(coh_l)*1.1
    ax.set_xlim([0,max_time])
    ax.set_ylim([0,max_coh])

    title_plot = 'Coherence Vs Convulsion Time (%) ' + self.brain_state_name
    self.indiv_plots_common(ax, title_plot, x_label='Convulsion time (%)')


  def plot_WT_average_KO_indiv_long(self):
    ax = self.indiv_fig()

    for n, Cxy in enumerate(self.all_KO_coh_data):
      if len(Cxy.long_line_plot_1rec_m) > 0:
        ax.plot(self.f_array, Cxy.long_line_plot_1rec_m, label = str(n), linewidth=0.5)

    ax.plot(self.f_array, self.mean_WT_lon_plot_line, label = 'long-range WT', color='black')
    ax.fill_between(self.f_array, np.array(self.mean_WT_lon_plot_line) - np.array(self.sem_WT_lon_plot_line),
                        np.array(self.mean_WT_lon_plot_line) + np.array(self.sem_WT_lon_plot_line),
                        edgecolor='#bbbbbb', facecolor= '#bbbbbb',
                        linewidth=1, linestyle='dashed', antialiased=False)

    title_plot = 'Mean Long-range KO vs WT(i) coherences ' + self.brain_state_name
    self.indiv_plots_common(ax, title_plot)


  def indiv_fig(self):
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    return ax

  def indiv_plots_common(self, ax, title_plot, y_label='Average $z^{-1}$ Coherence', x_label='frequency [Hz]'):

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(title_plot, fontsize=18)
    #self.rest_of_the_plot()
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel(x_label, fontsize=18)
    plt.show()

my_coherence = coherence_eeg()
ind_calc = [] # list of indiv_tests classes

if __name__=="__main__":
     app = QApplication(sys.argv)
     w = MyForm()
     w.show

     sys.exit(app.exec())
