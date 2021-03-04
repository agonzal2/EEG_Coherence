import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

folder = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/TAINI CDKL5 Recordings/CDKL5_1960/' # '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/'
npy_prefix = 'CDKL51960'
dat_file = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/TAINI CDKL5 Recordings/CDKL5_1960/TAINI_1047_B_CDKL5_1960_Redo-2020_12_10-0000.dat'

montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
#brain_states_file = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/S7062_BL1_States.xls'
brain_states_file = ''
first_sample = 1
n_electrodes = 16
amp_filter = 4000 # in Taini values that are 32.755 are lost samples

prm.set_sampling_rate(250.41)
sr=prm.get_sampling_rate()
downsampling = 2

if brain_states_file == '' :
  v_all = load_16_EEG_taini_down(dat_file, downsampling, amp_filter, first_sample, sr)
else: 
  brain_states = import_brain_states(brain_states_file)
  v_all, v_wake, v_nrem, v_rem, v_conv = load_16_EEG_taini_down_by_state(dat_file, brain_states, downsampling, amp_filter, first_sample, sr)
  f_wake = npy_prefix + '_wake' + '.npy'
  f_nrem = npy_prefix + '_nrem' + '.npy'
  f_rem = npy_prefix + '_rem' + '.npy'
  f_conv = npy_prefix + '_conv' + '.npy'
  f_all = npy_prefix + '.npy'
  os.chdir(folder)
  np.save(f_wake, v_wake)
  np.save(f_nrem, v_nrem)
  np.save(f_rem, v_rem)
  np.save(f_conv, v_conv)

f_all = npy_prefix + '.npy'
os.chdir(folder)
np.save(f_all, v_all)




#colors=dict(mag = 'darkblue', grad='b', eeg='k', ecg='m', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

#plt.plot(custom_raw._data[6])

#plt.plot(custom_raw._data[0])
#plt.plot(custom_raw._data[11])
#plt.show()

