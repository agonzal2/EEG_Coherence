import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

folder = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/'
npy_prefix = 'S7062Bas1'
dat_file = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/TAINI_1036_S7062-2_A-2020_01_13-0000.dat'

montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
brain_states_file = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/S7062_BL1_States.xls'
first_sample = 14580001
n_electrodes = 16
amp_filter = 4000

prm.set_sampling_rate(250.41)
sr=prm.get_sampling_rate()
downsampling = 2

brain_states = import_brain_states(brain_states_file)

v_all, v_wake, v_nrem, v_rem, v_conv = load_16_EEG_taini_down_by_state(dat_file, brain_states, downsampling, amp_filter, first_sample, sr)

f_all = npy_prefix + '.npy'
f_wake = npy_prefix + '_wake' + '.npy'
f_nrem = npy_prefix + '_nrem' + '.npy'
f_rem = npy_prefix + '_rem' + '.npy'
f_conv = npy_prefix + '_conv' + '.npy'

os.chdir(folder)
np.save(f_all, v_all)
np.save(f_wake, v_wake)
np.save(f_nrem, v_nrem)
np.save(f_rem, v_rem)
np.save(f_conv, v_conv)




#colors=dict(mag = 'darkblue', grad='b', eeg='k', ecg='m', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

#plt.plot(custom_raw._data[6])

#plt.plot(custom_raw._data[0])
#plt.plot(custom_raw._data[11])
#plt.show()

