import sys
import os
import glob
import re as re
import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_ind
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
    d = os.getcwd() + "\\"
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
      self.compareGroups('ShortKO', 'ShortWT')
      self.write_table_results(3)
      self.sheet_name = sheet_n + "Long"
      self.compareGroups('LongKO', 'LongWT')
      self.write_table_results(4)
      self.print2pdf(file_name)
      self.closeFigures()

    self.workbook.close()
    print ("Data and figures correctly exported to excel")

  def compareGroups(self, group1, group2):

    df_g1 = self.dfs[group1] # KO
    df_g2 = self.dfs[group2] # WT

    df_g1.reset_index(inplace=True) # the pandas has a multiindex, including the Freqs column.
    df_g2.reset_index(inplace=True)
    # Deleting all that information that we do not need
    df_g1.drop(['index', 'Unnamed: 0', 'Freqs'], axis=1, inplace=True)
    df_g2.drop(['index', 'Unnamed: 0', 'Freqs'], axis=1, inplace=True)

    freq_l =  df_g1.index.values # df_g2[0].tolist()
    df_means = pd.DataFrame(freq_l)
    df_means.columns = ['Frequency (Hz)']
    #df_g2.drop(df_g2.columns[0], axis=1, inplace=True)
    new_cols = []
    for i in np.arange(len(df_g1.columns)):
      new_cols.append("n"+str(i + 1 + len(df_g2.columns)))

    df_g1.columns = new_cols

    df_to_suff = df_g2.join(df_g1) # dataframe with all the coherence to shuffle

    # plotting the means along frequency
    column1 = '$Syngap^{+/-\u0394 GAP}$'
    column2 = '$Syngap^{+/+}$'
    df_means[column1] = df_g1.mean(axis=1)
    df_means[column2] = df_g2.mean(axis=1)
    colors = ['black', '#0836a9']
    if 'abs' in self.coh_type:
      dx = df_means.plot(x='Frequency (Hz)', y =[column2, column1], ylim = (0,1), color = colors)
    else:
      dx = df_means.plot(x='Frequency (Hz)', y =[column2, column1], color = colors)
    dx.set_ylabel('Mean $z^{-1}$ Coherence')
    plt.show()

    self.freq_list_results = []
    self.get_frequency_bands()

    labels = []
    mean_KO_l = []
    mean_WT_l = []
    sem_WT_l = []
    sem_KO_l = []

    num_bands = len(self.freq_list)
    pvalues = np.zeros(num_bands)

    # creating a list with the real differences for each freq band
    real_diff = []
    l_df_freqs = []
    for freq_band in self.freq_list:
        f_1, f_2 = self.obtainfreqs(freq_band)
        df_freqs = df_to_suff.iloc[f_1:f_2]

        real_averages = list(df_freqs.mean(axis=0))
        l_df_freqs.append(df_freqs)
        real_diff.append(calc_perm_test(real_averages))

    # Looping over all the permutations.
    max_distribution = [] # list of the most extreme diff value of each iteration
    for i in range(self.ui.BoxNumberShuffles.value()):
      # For each permutation, we choose the most extreme result among all the bands
      # to build the multiple comparison distribution
      max_candidates = []
      for n_band, freq_band in enumerate(self.freq_list):

          cols = list(l_df_freqs[n_band].columns)
          cols_s = random.sample(cols, len(cols))
          df2 = l_df_freqs[n_band][cols_s]
          df2.columns = list(df_freqs.columns)
          averages = list(df2.mean(axis=0))
          diff_perm = calc_perm_test(averages) # returns difference between the permutated groups
          if diff_perm > real_diff[n_band]:
              pvalues[n_band] = pvalues[n_band] + 1
          max_candidates.append(diff_perm)

      max_distribution.append(max(max_candidates))

    # plotting max distribution
    fig1 = plt.figure()
    bx = fig1.add_subplot(111)
    a_threshold_dist = np.asarray(max_distribution)
    perc95 = np.percentile(a_threshold_dist, 95)
    n, bins, patches = bx.hist(a_threshold_dist, bins=20, alpha = 0.05, color='black')
    hist_heights = []
    for p in patches:
      hist_heights.append(p._height)
    max_height = max(hist_heights)
    bx.plot([perc95, perc95], [0,max_height], color = '#888888', linestyle='dashed', lw=1)
    plt.show

    # Calculating pvalues, without correction and exporting to excel
    worksheet_ti= self.workbook.add_worksheet(self.sheet_name)
    header = 'Correction for multiple comparisons: if the difference is bigger than the 95 percentile threshold ('+ str(perc95) +') then there is significance'
    worksheet_ti.write(0,0,header)
    worksheet_ti.write(1, 0, "freq band")
    worksheet_ti.write(1, 1, "Freq_1")
    worksheet_ti.write(1, 2, "Freq_2")
    worksheet_ti.write(1, 3, "p_value")
    worksheet_ti.write(1, 4, "difference")
    for n_band, freq_band in enumerate(self.freq_list):
      p_val = pvalues[n_band] / self.ui.BoxNumberShuffles.value()
      self.freq_list_results.append(p_val)
      #print('The p-value for freqs between ', freq_1, ' and ', freq_2, 'Hz is: ', p_val)
      worksheet_ti.write(n_band + 2, 0, freq_band[0])
      worksheet_ti.write(n_band + 2, 1, freq_band[1])
      worksheet_ti.write(n_band + 2, 2, freq_band[2])
      worksheet_ti.write(n_band + 2, 3, p_val)
      worksheet_ti.write(n_band + 2, 4, real_diff[n_band])

    # plotting bars graph
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
    ax.set_ylabel('Mean $z^{-1}$ Coherence')
    if 'abs' in self.coh_type:
      ax.set_ylim([0,1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    fig.tight_layout()
    plt.show()

    # Once the figures are plotted, include them in the excel sheet
    cell_positions = ('G3', 'O3', 'W3','G30', 'O30', 'W30')
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

  def write_table_results(self, position):

    for n, freq_interval_results in enumerate(self.freq_list_results):
      self.ui.tableFrequencies.setItem(n+1, position, QTableWidgetItem(str(freq_interval_results)))

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
