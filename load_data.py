import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

folder = '/media/jorge/DATADRIVE0/Data/Alfredo/Syngap/KOTest/'


montage_name = '/media/jorge/DATADRIVE0/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'
n_electrodes = 32

prm.set_sampling_rate(1000)
sr=prm.get_sampling_rate()
downsampling = 8

l_folders = []
l_xls_files = []
l_numpy_files = []

# dividing files by animal
os.chdir(folder)
d = os.getcwd() + '/'
matching_files = glob.glob(r'*xls')
for matching_file in matching_files:
    l_xls_files.append(d+matching_file)
    l_folders.append(d+ matching_file.replace('_States.xls', '') + '/')
    l_numpy_files.append(d+ matching_file.replace('_States.xls', '') + '_Down'+ str(downsampling) )    

for i, xls_file in enumerate(l_xls_files):
    # First extract data for the whole recording but just for the NonREM state
    brain_states = import_brain_states(xls_file)
    v_all, v_wake, v_nrem, v_rem, v_conv = load_32_EEG_downsampled_bystate(l_folders[i], montage_name, '100', brain_states, downsampling)
    f_all = l_numpy_files[i] + '.npy'
    f_wake = l_numpy_files[i] + '_wake' + '.npy'
    f_nrem = l_numpy_files[i] + '_nrem' + '.npy'
    f_rem = l_numpy_files[i] + '_rem' + '.npy'
    f_conv = l_numpy_files[i] + '_conv' + '.npy'
    np.save(f_all, v_all)
    np.save(f_wake, v_wake)
    np.save(f_nrem, v_nrem)
    np.save(f_rem, v_rem)
    np.save(f_conv, v_conv)



