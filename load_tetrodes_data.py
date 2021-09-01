import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import mne
import os
import glob

from initial_processes import *

recording_folder = '2021-04-29_10-33-40'
root_folder = '/media/jorge/otherprojects/Data/Alfredo/tetrodes/'
folder_name = root_folder + recording_folder + '/'

montage_name = '/media/jorge/otherprojects/Code/coherence/EEG_Coherence/standard_16tetrodes.elc'
n_electrodes = 16
amp_filter = 750

prm.set_sampling_rate(1000)
sr=prm.get_sampling_rate()
downsampling = 8

npy_prefix = root_folder + 'npy/' + recording_folder + '_' + 'Downs' + str(downsampling)


# dividing files by animal
v1, v2, v3, v4 = load_16channel_tetrode(folder_name, montage_name, downsampling, amp_filter)
f_1 = npy_prefix + '_1.npy'
f_2 = npy_prefix + '_2.npy'
f_3 = npy_prefix + '_3.npy'
f_4 = npy_prefix + '_4.npy'
np.save(f_1, v1)
np.save(f_2, v2)
np.save(f_3, v3)
np.save(f_4, v4)
        


