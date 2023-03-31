import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
import os
import glob

from initial_processes import *

# Parameters
prm.set_sampling_rate(2000)
sr=prm.get_sampling_rate()
downsampling = 8

recordings_folder = '/media/prignane/T7/Sleep_Data_For_Jorge/Raw_data/'  #/Rat688/session1/all_channels/'
path_info_file = recordings_folder + 'sleep_channel_info.xlsx'

# create a subfolder for the npy data
if not os.path.exists(recordings_folder + 'npy_data/'):
    os.makedirs(recordings_folder + 'npy_data/')
states = ['active', 'rest', 'noREM', 'REM', 'seizures']

# loading info file that contains the location and metadeta of the recordings
df_info = pd.read_excel(path_info_file)
l_df_info =  df_info.values.tolist()
ids = [x[0] for x in l_df_info]
genotypes = [x[1] for x in l_df_info]
recording_folders = [x[2] for x in l_df_info]
source_numbers = [str(x[3]) for x in l_df_info] # there are numbers passed as strings in the list
chosen_electrodes = [x[4:] for x in l_df_info]

# creating states subfolders for the data (brain states and genotypes)
for state in states:
  if not os.path.exists(recordings_folder + 'npy_data/' + state):
    os.makedirs(recordings_folder + 'npy_data/' + state)
  for geno in genotypes:
    if not os.path.exists(recordings_folder + 'npy_data/' + state + '/' + geno):
      os.makedirs(recordings_folder + 'npy_data/' + state + '/' + geno)

# looping over the recordings
for f_idx, folder in enumerate(recording_folders):

  # creating a list with the brain states (Active, Rest, NonREM, REM)
  d_brain_states = {0: [], 1: [], 2: [], 3: [], 4:[]}
  recording_folder = recordings_folder + folder + '/'
  df_states = pd.ExcelFile(recording_folder + folder + '.xlsx')
  d_brain_states[0] = pd.read_excel(df_states, 'active')
  d_brain_states[1] = pd.read_excel(df_states, 'rest')
  d_brain_states[2] = pd.read_excel(df_states, 'noREM')
  d_brain_states[3] = pd.read_excel(df_states, 'REM')
  d_brain_states[4] = pd.read_excel(df_states, 'seizures')


  # calling the function to split the data into the different states and downsample it
  v_active, v_rest, v_nonrem, v_rem, v_seizures = load_16_lfp_downsampled(recording_folder, source_numbers[f_idx], sr, downsampling, chosen_electrodes[f_idx], d_brain_states)

  # saving the data
  f_active = recordings_folder + 'npy_data/' + states[0] + '/' + genotypes[f_idx] + '/' + folder + '_' + states[0] + '_down_' + str(downsampling) +'.npy'
  f_rest = recordings_folder + 'npy_data/' + states[1] + '/' + genotypes[f_idx] + '/' + folder + '_' + states[1] + '_down_' + str(downsampling) +'.npy'
  f_nonrem = recordings_folder + 'npy_data/' + states[2] + '/' + genotypes[f_idx] + '/' + folder + '_' + states[2] + '_down_' + str(downsampling) +'.npy'
  f_rem = recordings_folder + 'npy_data/' + states[3] + '/' + genotypes[f_idx] + '/' + folder + '_' + states[3] + '_down_' + str(downsampling) +'.npy'
  f_seizures = recordings_folder + 'npy_data/' + states[4] + '/' + genotypes[f_idx] + '/' + folder + '_' + states[4] + '_down_' + str(downsampling) +'.npy'

  np.save(f_active, v_active)
  np.save(f_rest, v_rest)
  np.save(f_nonrem, v_nonrem)
  np.save(f_rem, v_rem)
  np.save(f_seizures, v_seizures)

  
  

