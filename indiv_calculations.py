
from scipy.signal import decimate
import numpy as np
from initial_processes import *

'Dictionary for color of traces'
colors=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='g', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

class indiv_tests ():
  
  def __init__(self, recording_path, recording_index, sample_rate):
    self.recording_path = recording_path
    self.recording_index = recording_index
    self.sample_rate = sample_rate

  
  def load_npy32openephys(self, montage_name):
    self.rawdata = npy32mne(self.recording_path, montage_name, self.sample_rate)    


  def load_npy16taini(self, montage_name):
    self.rawdata = taininumpy2mne(self.recording_path, montage_name, self.sample_rate)

  
  def load_npy16tetrodes(self, montage_name):
    self.rawdata = tetrodesnumpy2mne(self.recording_path, montage_name, self.sample_rate)

  
  def apply_ica(self):
    # https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA
    a = 0
  
  def downsample(self, k_down = 1):    
    if (k_down > 1): 
      print("Downsampling")
      self.raw_data = self.rawdata.copy().resample(int(1000/k_down), npad='auto')
    else:
      self.raw_data = self.rawdata
    del self.rawdata


  def bandpass(self, lf, hf, electrodes, njobs):
    filtered_data = self.rawdata.filter(lf, hf, electrodes, n_jobs = njobs)
    filtered_data.plot(scalings = "auto", order=electrodes, show_options = "true")
    mne.viz.plot_raw_psd(filtered_data, fmin=0, fmax=100, picks=electrodes)
  
  def plotRawData(self, binsize, tmin, electrodes):
    #if len(electrodes) < 8: 
    #  plotnumber = len(electrodes)
    #else:
    #  plotnumber = 8

    fig = self.rawdata.plot(None, binsize, tmin, len(electrodes), color = colors, scalings = "auto", order=electrodes, show_options = "true" )
    #fig.add_axes(rect=(0.2,0.2,0.2,0.2))
    #fig(num=1, figsize=(8, 6))

  def plotPS(self, tmin=0, tmax=60, electrodes=[0,1,2]):
    ## https://mne.tools/stable/generated/mne.viz.plot_raw_psd.html
    freq = (self.sample_rate // 50)*25  # plots different frequency axis depending on the sampling rate.
    # Makes sure that the n_fft is a power of 2 and equal or smaller than the number of samples
    # https://stackoverflow.com/questions/29439888/what-is-nfft-used-in-fft-function-in-matlab/29440071
    total_samples = (tmax - tmin) * self.sample_rate
    size_fft = power_of_two(total_samples)
    if size_fft > 2048:
      size_fft = 2048
    mne.viz.plot_raw_psd(self.rawdata, fmin=0, fmax = freq, tmin=tmin, tmax=tmax, n_fft = size_fft, picks=electrodes)

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

