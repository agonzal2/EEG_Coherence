import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

npy_file = '/media/jorge/DATADRIVE0/Data/Alfredo/Taini/S7062/S7062Bas1_rem.npy'
#montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_16grid_taini1.elc'
montage_name = '/media/jorge/DATADRIVE0/Code/coherence/EEG_Coherence/standard_4grid_taini1.elc'
channels_list=[2,11,14,15]

prm.set_sampling_rate(250.41) # data downsampled by 8
sample_rate = prm.get_sampling_rate()

'Dictionary for color of traces'
#colors=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='g', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

#raw_data = taininumpy2mne(npy_file, montage_name, sample_rate/2)
raw_data = taininumpy2mnechannels(npy_file, montage_name, sample_rate/2, channels_list)

#raw_data.plot(None, 5, 20, 8,color = colors, scalings = "auto", order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,
#                                           14,15,16,32,17,18,19,20,21,22,
#                                           23,24,25,26,27,28,29,30,31,32], show_options = "true" )

raw_data.plot(scalings = "auto")
                                        
raw_data.plot_psd(average=True)

plt.show()

a = 2