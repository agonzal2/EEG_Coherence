import sys
import os
import glob
import re as re
import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_ind, sem
from statistics import mean
import matplotlib.pyplot as plt
import xlsxwriter

from w_perm_analysis import *

from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication, QFileDialog, QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QPushButton, QErrorMessage
from PyQt5.QtCore import pyqtSlot


import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from datetime import datetime


def calc_perm_test(perm_averages):
    half_len = int(len(perm_averages)/2)
    avg_sec = []
    avg_first = []
    for n, avgs in enumerate(perm_averages):
        if n < half_len:
            avg_first.append(avgs)
        else:
            avg_sec.append(avgs)

    diff_perm = mean(avg_sec) - mean(avg_first)
    return diff_perm

def get_metadata(filename):
  '''
    obtains metadata from the name of a file
    input : filename
    returns three strings: brain state, long_distance and kind of coherence, absolute or imaginary

  '''
  if "_REM" in filename:
        brain_state = "REM"
  elif "NonREM" in filename:
        brain_state = "NonREM"

  for distance in np.arange(1, 6, 0.5):
    if str(distance)[:3] in filename:
      long_distance = str(distance)[:3]
      break

  if "abs" in filename:
    coh_type = "abs"
  else:
    coh_type = "imag"

  return brain_state, long_distance, coh_type



class MyForm(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_w_perm_Analysis()
    self.ui.setupUi(self)

    self.ui.ButtonSelectFolder.clicked.connect(self.selectFolder)
    self.ui.ButDelFile.clicked.connect(self.delFile)
    self.ui.ButtonRunAll.clicked.connect(self.runAll)
    self.ui.ButtonClearFigures.clicked.connect(self.closeFigures)
    self.ui.Button2PDF.clicked.connect(self.print2pdf)
    self.ui.Button2PNG.clicked.connect(self.print2png)

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
    os.chdir(str(self.ui.labelFileSelected.text()))
    d = os.getcwd() + "/" # "\\" for Windows
    matching_files = glob.glob(r'*xlsx')
    l_xlsx_files = []
    self.workbook = xlsxwriter.Workbook('permutation_stats.xlsx')
    for matching_file in matching_files:
      file_n = re.sub('.xlsx', '', matching_file)
      file_name = 'perm_analysis_' + file_n
      self.brain_state, self.longD, self.coh_type = get_metadata(file_n)
      sheet_n = self.brain_state + self.longD + self.coh_type
      file_route = d + matching_file
      l_xlsx_files.append(file_route)

      self.dfs = pd.read_excel(file_route, sheet_name=None)
      self.sheet_name = sheet_n + "Short"
      self.pvalue_position = 3
      self.compareGroups('ShortKO', 'ShortWT')
      self.sheet_name = sheet_n + "Long"
      self.pvalue_position = 4
      self.compareGroups('LongKO', 'LongWT')
      self.print2pdf(file_name)
      self.closeFigures()

    self.workbook.close()
    print ("Data and figures correctly exported to excel")

  def compareGroups(self, group1, group2):

    df_g1 = self.dfs[group1] # KO
    df_g2 = self.dfs[group2] # WT

    df_g1.reset_index(inplace=True) # the pandas has a multiindex, including the Freqs column.
    df_g2.reset_index(inplace=True)
    freq_samples = df_g1['Freqs'].to_numpy()
    # Deleting all that information that we do not need
    df_g1.drop(['index', 'Unnamed: 0', 'Freqs'], axis=1, inplace=True)
    df_g2.drop(['index', 'Unnamed: 0', 'Freqs'], axis=1, inplace=True)

    freq_l =  df_g1.index.values # df_g2[0].tolist()
    df_means = pd.DataFrame(freq_l)
    df_means.columns = ['Frequency (Hz)']
    new_cols = []
    for i in np.arange(len(df_g1.columns)):
      new_cols.append("n"+str(i + 1 + len(df_g2.columns)))

    df_g1.columns = new_cols

    df_to_suff = df_g2.join(df_g1) # dataframe with all the coherence to shuffle

    # plotting the means along frequency
    mean_ko = df_g1.mean(axis=1).to_numpy()
    mean_wt = df_g2.mean(axis=1).to_numpy()
    sem_ko = sem(df_g1, axis=1)
    sem_wt = sem(df_g2, axis=1)
    fig0 = plt.figure()
    dx = fig0.add_subplot(111)
    dx.plot(freq_samples, mean_ko, color = '#0836a9', label = '$Syngap^{+/-\u0394 GAP}$')
    dx.plot(freq_samples, mean_wt, color = 'black', label = '$Syngap^{+/+}$')
    dx.fill_between(freq_samples, np.array(mean_ko) - np.array(sem_ko), np.array(mean_ko) + np.array(sem_ko),
                        edgecolor='#b4cdec', facecolor= '#b4cdec', linewidth=1, antialiased=False) #linestyle='dashed',
    dx.fill_between(freq_samples, np.array(mean_wt) - np.array(sem_wt), np.array(mean_wt) + np.array(sem_wt),
                        edgecolor='#bbbbbb', facecolor= '#bbbbbb', linewidth=1, antialiased=False)

    if 'abs' in self.coh_type:
      dx.set_ylim(0,1)
    else:
      maxko = np.amax(mean_ko + sem_ko)
      maxwt = np.amax(mean_wt + sem_wt)
      dx.set_ylim(0,max(maxko, maxwt))
    dx.set_xlabel('Frequency (Hz)')
    dx.set_ylabel('Mean $z^{-1}$ ' + self.coh_type + ' Coherence')
    dx.legend()
    plt.title(self.sheet_name)
    plt.show()

    # permutation analysis
    matrix_coh = df_to_suff.to_numpy() # dataframe to numpy to make it faster
    matrix_coh_t = np.transpose(matrix_coh) # traspose to average per frequencies
    n, frequencies = np.shape(matrix_coh_t) # nxm matrix. n = number of animals

    averages_1 = np.mean(matrix_coh_t[:int(n/2),:], axis = 0) # averages for every 0.5Hz for the first animals
    averages_2 = np.mean(matrix_coh_t[int(n/2):,:], axis = 0) # averages for every 0.5Hz for the second animals
    real_diferences = averages_1 - averages_2 # these are the differences we will need to compare to the permutations

    anim = int(n/2)
    pvalues = np.zeros(np.size(averages_1)).astype(int) # where we are gonna sum the pvalues
    shuffled_coh = matrix_coh_t
    # loop over the total number of permutations
    for i in np.arange(self.ui.BoxNumberShuffles.value()):
      np.random.shuffle(shuffled_coh) # stores the shuffled data in the same nd array
      shuf_aver_1 = np.mean(shuffled_coh[:anim,:], axis = 0) # averages for every 0.5Hz for the first animals
      shuf_aver_2 = np.mean(shuffled_coh[anim:,:], axis = 0) # averages for every 0.5Hz for the second animals
      perm_diffs = shuf_aver_1 - shuf_aver_2
      pvalues = pvalues + np.ceil(real_diferences - perm_diffs)

    pvalues = pvalues/self.ui.BoxNumberShuffles.value() # final pvalues for that comparison

    # plotting permutation pvalues
    fig1 = plt.figure()

    bx = fig1.add_subplot(111)
    bx.set_ylabel('pvalue')
    bx.set_xlabel('frequency (Hz)')
    title = "pvalues " + self.sheet_name
    thresh_min = 0.025*np.ones(len(freq_samples))
    thresh_max = 0.975*np.ones(len(freq_samples))
    bx.plot(freq_samples, thresh_min, color = 'red', linewidth=1, linestyle='dashed')
    bx.plot(freq_samples, thresh_max, color = 'red', linewidth=1, linestyle='dashed')
    bx.plot(freq_samples,pvalues)
    bx.set_ylim(0,1)
    plt.title(title)
    plt.show

    self.freq_list_results = []
    self.get_frequency_bands()

    # Calculating pvalues, without correction and exporting to excel
    worksheet_ti= self.workbook.add_worksheet(self.sheet_name)
    worksheet_ti.write(1, 0, "Frequency")
    worksheet_ti.write(1, 1, "Freq_1")
    worksheet_ti.write(1, 2, "Freq_2")
    worksheet_ti.write(1, 3, "p_value")
    worksheet_ti.write(1, 4, "difference")
    for n_band, freq_band in enumerate(freq_samples):
      worksheet_ti.write(n_band + 2, 0, freq_band)
      worksheet_ti.write(n_band + 2, 3, pvalues[n_band])


    # pvalues for the bar graph
    for freq_band in self.freq_list:
      f_1, f_2 = self.obtainfreqs(freq_band)
      pvalue = 0 # where we are gonna sum the pvalues
      df_bands = df_g2.join(df_g1) # needs to do it every time, if not the shuffling affects future iterations
      matrix_band_coh = df_bands.to_numpy() # dataframe to numpy to make it faster
      matrix_band_coh_t = np.transpose(matrix_band_coh) # traspose to average per frequencies
      shuffled_band_coh = matrix_band_coh_t[:, f_1:f_2] # np.mean(matrix_coh_t[:, f_1:f_2], axis=1)
      first_animals = shuffled_band_coh[:anim, :]
      average_1 = np.mean(shuffled_band_coh[:anim, :])
      average_2 = np.mean(shuffled_band_coh[anim:, :])
      real_diference = average_1 - average_2
      # loop over the total number of permutations
      for i in np.arange(self.ui.BoxNumberShuffles.value()):
        np.random.shuffle(shuffled_band_coh) # stores the shuffled data in the same nd array
        shuf_aver_1 = np.mean(shuffled_band_coh[:anim, :])
        shuf_aver_2 = np.mean(shuffled_band_coh[anim:, :])
        perm_diff = shuf_aver_1 - shuf_aver_2
        pvalue = pvalue + np.ceil(real_diference - perm_diff)

      pvalue = pvalue/self.ui.BoxNumberShuffles.value() # final pvalues for that comparison
      self.freq_list_results.append(pvalue)

    self.write_table_results()


    # plotting bars graph
    labels = []
    mean_KO_l = []
    mean_WT_l = []
    sem_WT_l = []
    sem_KO_l = []
    column1 = '$Syngap^{+/-\u0394 GAP}$'
    column2 = '$Syngap^{+/+}$'
    df_means[column1] = df_g1.mean(axis=1)
    df_means[column2] = df_g2.mean(axis=1)
    for freq_band in self.freq_list:
      f_1, f_2 = self.obtainfreqs(freq_band)

      labels.append(freq_band[0])
      mean_KO_l.append(float(df_means[[column1]].iloc[range(f_1,f_2,1)].mean(axis=0)))
      mean_WT_l.append(float(df_means[[column2]].iloc[range(f_1,f_2,1)].mean(axis=0)))
      sem_KO_l.append(float(df_means[[column1]].iloc[range(f_1,f_2,1)].sem(axis=0)))
      sem_WT_l.append(float(df_means[[column2]].iloc[range(f_1,f_2,1)].sem(axis=0)))
    width = 0.35
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, mean_WT_l, width, yerr=sem_WT_l, capsize = 3, label='$Syngap^{+/+}$', color='black')
    rects2 = ax.bar(x + width/2, mean_KO_l, width, yerr=sem_KO_l, capsize = 3, label='$Syngap^{+/-\u0394 GAP}$', color= '#0836a9')
    ax.set_ylabel('Mean $z^{-1}$ ' + self.coh_type + ' Coherence')
    if 'abs' in self.coh_type:
      ax.set_ylim([0,1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    fig.tight_layout()
    plt.show()

    # Once the figures are plotted, include them in the excel sheet
    cell_positions = ('G3', 'Q3', 'AA3','G30', 'Q30', 'AA30')
    figFolder = os.getcwd()
    if figFolder:
        prefix = '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        j = 0
        for i in plt.get_fignums():
            plt.figure(i)
            file_name = figFolder + prefix +'figure%d.png' % i
            plt.savefig(file_name)
            worksheet_ti.insert_image(cell_positions[j], file_name)
            j = j+1
    else:
        self.error_msg.showMessage("Something went wrong saving the figure")

  def obtainfreqs(self, freq_band):
    freq_1 = freq_band[1]
    freq_2 = freq_band[2]
    f_1 = freq_1*2 -2 # fitting between the frequencies and the indexes of the df
    f_2 = freq_2*2 -2
    return f_1, f_2

  def get_frequency_bands(self):
    nrows = self.ui.tableFrequencies.rowCount()
    self.freq_list = [] # list of 6 element tuples, frequency bands and the results for those bands

    for row in range(1, nrows):
      band_name = self.ui.tableFrequencies.item(row, 0).text()

      if band_name == "":
        print (row - 1), 'frequency bands'
        break

      band_freq_from = self.ui.tableFrequencies.item(row, 1).text()
      band_freq_to = self.ui.tableFrequencies.item(row, 2).text()
      self.freq_list.append((band_name, int(band_freq_from), int(band_freq_to))) # double (()) is necessary

  def write_table_results(self):

    for n, freq_interval_results in enumerate(self.freq_list_results):
      self.ui.tableFrequencies.setItem(n+1, self.pvalue_position, QTableWidgetItem(str(freq_interval_results)))

  def closeFigures(self):
       plt.close('all')

  def print2pdf(self, filename = ''):
    if filename:
      filename2 = '/' + filename + '.pdf'
      pdf = matplotlib.backends.backend_pdf.PdfPages(str(self.ui.labelFileSelected.text()) + filename2)
      figs = [plt.figure(n) for n in plt.get_fignums()]
      for fig in figs:
        fig.savefig(pdf, format='pdf')
      pdf.close()
    else:
      self.error_msg.showMessage("It is necessary to select a folder")

  def print2png(self):
       figFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
       if figFolder:
            prefix = '/' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            for i in plt.get_fignums():
                 plt.figure(i)
                 plt.savefig(figFolder + prefix +'figure%d.png' % i)
       else:
            self.error_msg.showMessage("It is necessary to select a folder")



if __name__=="__main__":
     app = QApplication(sys.argv)
     w = MyForm()
     w.show

     sys.exit(app.exec())
