
from scipy.signal import decimate
import numpy as np
from initial_processes import *

'Dictionary for color of traces'
colors=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='g', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

class indiv_tests ():
  
  def __init__(self, recording_path, recording_index):
    a = 0
    self.recording_path = recording_path
    self.recording_index = recording_index

  def load_recordings(self, montage_name, k_down = 1):

    rawdata = load_32_EEG(self.recording_path, montage_name, '100')
    if (k_down > 1): 
      print("Downsampling")
      self.raw_data = rawdata.copy().resample(int(1000/k_down), npad='auto')
    else:
      self.raw_data = rawdata
    del rawdata
    #raw_times = np.ndarray.tolist(raw_data._times)
    #self.raw_data_downsampled = []
    #self.raw_times_downsampled = []

    #for i in np.arange(32):
    #  self.raw_data_downsampled.append(decimate(raw_data[i,:], k_down))
    #self.raw_times_downsampled = decimate(raw_times[i,:], k_down)

    #del raw_data
    #del raw_times

  
  def plotRawData(self, binsize, tmin, electrodes):
    if len(electrodes) < 8: 
      plotnumber = len(electrodes)
    else:
      plotnumber = 8

    self.raw_data.plot(None, binsize, tmin, plotnumber, color = colors, scalings = "auto", order=electrodes, show_options = "true" )

  def plotPS(self, tmin=0, tmax=60, electrodes=[0,1,2]):
    ## https://mne.tools/stable/generated/mne.viz.plot_raw_psd.html
    mne.viz.plot_raw_psd(self.raw_data, fmin=0, fmax=100, tmin=tmin, tmax=tmax, picks=electrodes)

  def temporal_signal(self):
    a = 1

  def calc_fft(self):
    a = 2

  def calc_temporalPS(self):
    a = 3

  def calc_temporalEnergy(self):
    a = 4

  def calc_PS(self):
    a = 5

  def calc_coherogram(self):
    a = 6

