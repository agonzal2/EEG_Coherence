import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

# Variables that change every recording!!!
filename = 'TAINI_1036_S7087_Baseline1-2020_08_21-0000.dat' 
animal_id = 'S7087' 
baseline = 2  
first_sample = 39723457

# Variables that can change but not that often
folder = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/TAINI_Syngap-Acute_ETX_Recordings/' # '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/'
downsampling = 2
montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
n_electrodes = 16
amp_filter = 4000 # in Taini values that are 32.755 are lost samples
prm.set_sampling_rate(250.41)
sr=prm.get_sampling_rate()

npy_prefix = 'npy/' + animal_id + '_Baseline' + str(baseline) + '_' + 'Downs' + str(downsampling)
dat_file = folder + animal_id + '/' + filename
brain_states_file = folder + animal_id + '_BL' + str(baseline) + '_States-real_samp.xls'


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

