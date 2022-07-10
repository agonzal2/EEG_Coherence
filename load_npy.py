import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

def npy32mne(filename, montage_name):

    
    voltage_array = np.load(filename) # load
    
    if isinstance('montage_name', str):
        montage = mne.channels.read_custom_montage(montage_name)
    else:
        print("The montage name is not valid")

    channel_types=['eeg','eeg','eeg','eeg','eeg','eeg', 'eeg', 'eeg',
                   'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                   ,'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                   ,'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg', 'emg', 'emg', 'emg']

    info = mne.create_info(montage.ch_names, prm.get_sampling_rate(), ch_types=channel_types)

    'This makes the object that contains all the data and info about the channels.'
    'Computations like plotting, averaging, power spectrums can be performed on this object'

    custom_raw = mne.io.RawArray(voltage_array, info)
    del voltage_array
    return custom_raw



prm.set_sampling_rate(125) # data downsampled by 8

'Dictionary for color of traces'
colors=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='g', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

montage_name = '/media/jorge/DATADRIVE0/Code/MNE_Alfredo/standard_32grid_Alfredo.elc'
npy_file = '/media/jorge/DATADRIVE0/Data/Alfredo/Syngap/KOTest/S7025_D2-A_Down8_wake_conv.npy'

raw_data = npy32mne(npy_file, montage_name)

#raw_data.plot(None, 5, 20, 8,color = colors, scalings = "auto", order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,
#                                           14,15,16,32,17,18,19,20,21,22,
#                                           23,24,25,26,27,28,29,30,31,32], show_options = "true" )

raw_data.plot(scalings = "auto")
                                        
raw_data.plot_psd(average=True)

plt.show()

a = 2