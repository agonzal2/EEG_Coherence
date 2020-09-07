import sys
import numpy as np
import random
import pandas as pd
from scipy.stats import ttest_ind
from statistics import mean
import matplotlib.pyplot as plt

from w_perm_analysis import *

from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication, QFileDialog, QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QPushButton, QErrorMessage
from PyQt5.QtCore import pyqtSlot


import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

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

class MyForm(QMainWindow):
  def __init__(self):
    super().__init__()
    self.ui = Ui_w_perm_Analysis()
    self.ui.setupUi(self)

    self.ui.ButSelectFile.clicked.connect(self.selectFile)
    self.ui.ButDelFile.clicked.connect(self.delFile)
    self.ui.ButtonRunAll.clicked.connect(self.runAll)
    #self.ui.ButtonClearFigures.clicked.connect(self.closeFigures)
    #self.ui.Button2PDF.clicked.connect(self.print2pdf)
    #self.ui.Button2PNG.clicked.connect(self.print2png)

    self.freq_list_results = []

    # Error message (it is necessary to initialize it too)
    self.error_msg = QErrorMessage()
    self.error_msg.setWindowTitle("Error")
    self.show()


  def selectFile(self):
    file_name, _ = QFileDialog.getOpenFileName(self,
                                     'Select file',
                                     './',
                                     'Excel Files (*.xls *.xlsx)')
    if file_name:
      self.ui.labelFileSelected.setText(file_name)
    else:
      self.error_msg.showMessage("It is necessary to select a file")

  def delFile(self):
    self.ui.labelFileSelected.clear()

  def runAll(self):
    self.dfs = pd.read_excel(self.ui.labelFileSelected.text(), sheet_name=None, index_col=None, header=None)

    self.compareGroups('ShortKO', 'ShortWT')
    self.write_table_results(3)
    self.compareGroups('LongKO', 'LongWT')
    self.write_table_results(4)

  def compareGroups(self, group1, group2):

    df_g1 = self.dfs[group1] # KO
    df_g1.drop(df_g1.columns[0], axis=1, inplace=True)
    df_g2 = self.dfs[group2] # WT
    freq_l = df_g2[0].tolist()
    df_means = pd.DataFrame(freq_l)
    df_means.columns = ['Frequency (Hz)']
    df_g2.drop(df_g2.columns[0], axis=1, inplace=True)
    orig_cols = list(df_g1.columns)
    new_cols = []
    for i in orig_cols:
      new_cols.append(i+ len(df_g2.columns))

    df_g1.columns = new_cols

    df_to_suff = df_g2.join(df_g1) # dataframe with all the coherence to shuffle

    # plotting the means along frequency
    column1 = '$Syngap^{+/-\u0394 GAP}$'
    column2 = '$Syngap^{+/+}$'
    df_means[column1] = df_g1.mean(axis=1)
    df_means[column2] = df_g2.mean(axis=1)
    colors = ['black', '#0836a9']
    dx = df_means.plot(x='Frequency (Hz)', y =[column2, column1], ylim = (0,1), color = colors)
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

    for n_band, freq_band in enumerate(self.freq_list):
      p_val = pvalues[n_band] / self.ui.BoxNumberShuffles.value()
      self.freq_list_results.append(p_val)
      #print('The p-value for freqs between ', freq_1, ' and ', freq_2, 'Hz is: ', p_val)

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

    print('if the difference is bigger than the threshold (', perc95, ') then there is significance')
    for n_band, freq_band in enumerate(self.freq_list):
      print(freq_band[0], 'diff= ', real_diff[n_band])

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
    ax.set_ylim([0,1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()
    fig.tight_layout()
    plt.show()

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



if __name__=="__main__":
     app = QApplication(sys.argv)
     w = MyForm()
     w.show

     sys.exit(app.exec())
