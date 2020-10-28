import sys
import os.path
import numpy as np
from initial_processes import *
import parameters
import os
import matplotlib.pyplot as plt
prm = parameters.Parameters()
from OpenEphys import *
import mne
import xlrd

n_channels = 16
sample_rate = 1000
sample_datatype = 'int16'
display_decimation = 10

"Specifiy the start time and end times here!!!!"

start_time=600
end_time=900


fn="E:\\OneDrive - University of Edinburgh\\S7063 - Het BL1 and BL2\\TAINI_1033_S7063_Baseline1-2020_01_27-0000.dat"

dat_chans, t = parse_dat(fn, n_channels, sample_rate)



data=np.array(dat_chans)
# The emg electrodes are in the positions 1 and 14, we put them at the end
# to make the code easier later
# we also create extra emg channels to follow mne montage requirements
(original_elect, n_samples) = data.shape
final_data = np.zeros((19, n_samples))
final_data[0:14, :] = data[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],:]
final_data[14:16, :] = data[[1,14],:]

del(dat_chans)
del(data)

#datatp = final_data.transpose()
#del(final_data)

# do not need this
#time_axis, sub_data = sub_time_data(datatp, start_time, end_time, sample_rate)
#sub_datatp=sub_data.transpose()


montage_name = 'standard_16grid_taini1'
if isinstance('montage_name', str):
    montage = mne.channels.read_montage(montage_name)
else:
    print("The montage name is not valid")

# 14 eeg channels, 2 emg, and 3 that are required for mne compatibility
channel_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                   ,'eeg','eeg','eeg','eeg','eeg','eeg', 'emg', 'emg', 'emg', 'emg', 'emg']

'This creates the info that goes with the channels, which is names, sampling rate, and channel types.'
info = mne.create_info(montage.ch_names, prm.get_sampling_rate(), ch_types=channel_types,
                           montage=montage)

custom_raw = mne.io.RawArray(final_data, info)

'To do a basic plot below. The following can be added for specifc order of channels order=[4, 5, 3, 0, 1, 14, 15, 16]'
colors=dict(mag='darkblue', grad='b', eeg='k', eog='k', ecg='m',
     emg='g', ref_meg='steelblue', misc='k', stim='b',
     resp='k', chpi='k')

custom_raw.plot(None, 5, 20, 16,color = colors, scalings = "auto", order=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], show_options = "true" )#
