import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

folder = '/media/jorge/DATADRIVE0/Data/Alfredo/Syngap/KOTest/'

montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
brain_states_file = ''
first_sample = 1
last_sample = 2
n_electrodes = 16
amp_filter = 4000

prm.set_sampling_rate(250.41)
sr=prm.get_sampling_rate()
downsampling = 2

l_folders = []
l_xls_files = []
l_numpy_files = []


# dividing files by animal
os.chdir(folder)
d = os.getcwd() + '/'
"""matching_files = glob.glob(r'*xls')
for matching_file in matching_files:
    l_xls_files.append(d+matching_file)
    l_folders.append(d+ matching_file.replace('_States.xls', '') + '/')
    l_numpy_files.append(d+ matching_file.replace('_States.xls', '') + '_Down'+ str(downsampling) )   """  

#for i, xls_file in enumerate(l_xls_files):
# First extract data for the whole recording but just for the NonREM state
l_numpy_files = (brain_states_file.replace('_States.xls', '') + '_Down'+ str(downsampling) )
brain_states = import_brain_states(brain_states_file)
v_all, v_wake, v_nrem, v_rem, v_conv = load_16_EEG_taini_down_by_state(folder, brain_states, downsampling, amp_filter, first_sample)
f_all = l_numpy_files + '.npy'
f_wake = l_numpy_files + '_wake' + '.npy'
f_nrem = l_numpy_files + '_nrem' + '.npy'
f_rem = l_numpy_files + '_rem' + '.npy'
f_conv = l_numpy_files + '_conv' + '.npy'
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

