from scipy.signal import decimate
import numpy as np
from initial_processes import *

folder= raw_data = "D:\Data\Alfredo\Syngap\Test1R\S7025_D2-A"
montage_name = 'standard_32grid_Alfredo'

raw_data = load_32_EEG(folder, montage_name, '100')

raw_data.plot(None, 5, 20, 8, scalings = "auto", order=[0,1,2], show_options = "true", block = True)

print('Original sampling rate:', raw_data.info['sfreq'], 'Hz')