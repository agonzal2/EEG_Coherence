import os 
import numpy as np 
import scipy as sp
import pandas as pd
import scipy.signal
import scipy.io
import time
import struct
from copy import deepcopy 
import glob 

import matplotlib.pyplot as plt
import mne
import xlrd
from pathlib import Path
import parameters as prm

file_route = '/home/melissa/Cdkl5_Baseline/CD_05'
montage_name = '/home/melissa/Documents/taini_test/standard_16grid_taini1.elc'

def parse_dat(fn, number_of_channels, sample_rate):
    dat_raw = np.fromfile(fn, dtype='int16')
    step=16*1
    dat_chans=[dat_raw[c::step] for c in range(number_of_channels)]
    t=np.arange(len(dat_chans[0]),dtype=float)/sample_rate
    return dat_chans,t

def load_16_EEG_taini(file_route, montage_name):
    n_channels = 16
    sample_rate = 250.4
    
    os.chdir(file_route)
    d = os.getcwd() + '/'
    file_name = glob.glob(r'*dat')
    
    dat_chans, t=parse_dat(file_name[0], n_channels, sample_rate) 
    
    data=np.array(dat_chans)
    # the emg electrodes are in position 1 & 14, put them at the end. 
    # create extra emgchannels to follow mne montage requirement
    (original_elect, n_samples) = data.shape
    final_data = np.zeros((19, n_samples))
    final_data[0:14, :] = data[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],:]
    final_data[14:16, :] = data[[1,14],:]
    
    del(dat_chans)
    del(data)
    
    if isinstance(montage_name, str):
        montage = mne.channels.read_montage(montage_name)
    else:
        print("the montage name is not valid")
        
    #14 eeg channels, 2 emg, and 3 that are not required for mne compatability 
    channel_types=['eeg', 'emg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'emg', 'eeg']
    
    'This creates the info that goes with the channels, which is names, sampling rate and channel types.'
    info = mne.create_info(montage.ch_names, sample_rate, ch_types=channel_types, montage=montage)
    
    montage=mne.channels.read_montage(montage_name)
    
    custom_raw = mne.io.RawArray(final_data, info)
    
    return custom_raw

custom_raw=[]
custom_raw=load_16_EEG_taini(file_route, montage_name)

colors=dict(mag = 'darkblue', grad='b', eeg='k', ecg='m', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')

plt.plot(custom_raw._data[6])
plt.show()

#sample_data_folder = mne.datasets.sample.data_path(path = 'home/melissa/Cdkl5_Baseline/CD_05/')
#sample_data_raw_file = os.path.join(sample_data_folder, 'TAINI_1035_CD_05_Baseline1-2020_08_14-0000.dat')

data_path = '/home/melissa/Cdkl5_Baseline/CD_05/'
raw_fname = data_path + 'TAINI_1035_CD_05_Baseline1-2020_08_14-0000.dat'
tmin, tmax = 0, 20
raw = mne.io.read_raw_fif(raw_fname).crop(tmin, tmax).load_data()
tmin, tmax = 0, 20
#raw.plot(custom_raw.data[6])

#plot using the mne plot function rather than basic python matplotlib

#custom_raw.plot()#None, 5, 20, 11412, color=colors, scalings="auto", order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], show_options="true")
#amplitude filter, delete samples in the same sample of electrodes that exceeds a threshold amplitdue 

#for i in np.arange(33):
 #   state_voltage_array = state_voltage_array[:,abs(state_voltage_array[i+1,:]) amp]
 