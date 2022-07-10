from re import L
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

def npy32to6areas_mne(filename, montage_name, sampling_rate):
    """
    Load numpy array file of 32 electrodes + 3 aux
    and converts it to mne format in a montage with 6 areas

    filename: route to the .npy file
    montage_name: route to the mne montage file with 6 areas
    
    """
    l_custom_raw = []
    l_areas_positions = []
    l_voltages = []
    voltage_array = np.load(filename) # load
    
    l_areas_positions.append([0, 1, 8, 9, 10]) # middle_left
    l_areas_positions.append([2, 3, 4, 5, 6, 7]) # caudal_left
    l_areas_positions.append([11, 12, 13, 14, 15]) # frontal_left
    l_areas_positions.append([16, 17, 18, 19, 20]) # frontal_right
    l_areas_positions.append([24, 25, 26, 27, 28, 29]) # caudal_right
    l_areas_positions.append([21, 22, 23, 30, 31]) # middle_right

    areas_va = np.zeros(voltage_array[:len(l_areas_positions)].shape)
    for a, area in enumerate(l_areas_positions):
        area_va = np.zeros(voltage_array[:len(area)].shape)
        for pos, electrode in enumerate(area):
            area_va[pos] = voltage_array[electrode]
        areas_va[a] = area_va.mean(axis=0)


    if isinstance('montage_name', str):
        montage = mne.channels.read_custom_montage(montage_name)
    else:
        print("The montage name is not valid")

    channel_types=['eeg','eeg','eeg','eeg','eeg','eeg']
    
    info = mne.create_info(montage.ch_names, sampling_rate, ch_types=channel_types)

    'This makes the object that contains all the data and info about the channels.'
    'Computations like plotting, averaging, power spectrums can be performed on this object'

    custom_raw = mne.io.RawArray(areas_va, info)
    del areas_va
    del area_va
    del voltage_array
    return custom_raw



prm.set_sampling_rate(125) # data downsampled by 8

'Dictionary for color of traces'
colors=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m', emg='g', ref_meg='steelblue', misc='k', stim='b', resp='k', chpi='k')


montage_name = '/media/jorge/otherprojects/Code/coherence/EEG_Coherence/six_areas.elc'
npy_file = '/media/jorge/otherprojects/Data/Alfredo/Syngap/KOnumpy/convulsion/S7033_D3-A_Down8_conv.npy'

raw_data = npy32to6areas_mne(npy_file, montage_name, 125)

#raw_data.plot(None, 5, 20, 8,color = colors, scalings = "auto", order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,
#                                           14,15,16,32,17,18,19,20,21,22,
#                                           23,24,25,26,27,28,29,30,31,32], show_options = "true" )

raw_data.plot(scalings = "auto")
                                        
raw_data.plot_psd(average=True)

plt.show()

a = 2