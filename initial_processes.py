
"""
Created on Wed Nov 22 16:17:18 2017

@author: Alfredo Gonzalez-Sulser, University of Edinburgh
email: agonzal2@staffmail.ed.ac.uk
"""
from numpy import *
import pandas as pd
from scipy import spatial
from itertools import combinations
import parameters
import matplotlib.pyplot as plt
prm = parameters.Parameters()
from OpenEphys import *
import mne
import xlrd
import xlsxwriter
from openpyxl import load_workbook
import pathlib


def load_file(file):  #Opens text files.
    print(" Opening file " + file)

    data=loadtxt(file)
    print(len(data))
    return data


"The function below loads 16 channel headstage recordings individually, specify which one of a potential 4 to load"

def load_16channel_opto_individually(headstage_number):


    if headstage_number == 1:
        channels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

        'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
        data=loadFolderToArray(prm.get_filepath(), channels, chprefix = 'CH', dtype = float, session = '0', source = '100')#######load file
        data_adc=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'ADC', dtype = float, session = '0', source = '100')#######load file8

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,16]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

    if headstage_number == 2:
        channels=[17,18,19,20,21,22, 23,24,25,26,27,28,29,30,31,32]

        'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
        data=loadFolderToArray(prm.get_filepath(), channels, chprefix = 'CH', dtype = float, session = '0', source = '100')#######load file
        data_adc=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'ADC', dtype = float, session = '0', source = '100')#######load file8

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,16]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

    if headstage_number == 3:
        channels=[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]

        'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
        data=loadFolderToArray(prm.get_filepath(), channels, chprefix = 'CH', dtype = float, session = '0', source = '100')#######load file
        data_adc=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'ADC', dtype = float, session = '0', source = '100')#######load file8

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,16]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

    if headstage_number == 4:
        channels=[49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]

        'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
        data=loadFolderToArray(prm.get_filepath(), channels, chprefix = 'CH', dtype = float, session = '0', source = '100')#######load file
        data_adc=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'ADC', dtype = float, session = '0', source = '100')#######load file8

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,16]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.


    return data

def load_16channel_opto(headstage_number):

    'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
    data=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'CH', dtype = float, session = '0', source = '100')#######load file
    data_adc=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'ADC', dtype = float, session = '0', source = '100')#######load file8

    if headstage_number == 4:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,64]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.




    if headstage_number == 3:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,48]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.


    if headstage_number ==2:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,32]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

        n_channels=33


    if headstage_number ==1:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,16]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

    return data


"The function below loads the data in mne format, specify number of headstages"

def load_16_channel_opto_mne(headstage_number):


    'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
    data=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'CH', dtype = float, session = '0', source = '100')#######load file
    data_adc=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'ADC', dtype = float, session = '0', source = '100')#######load file8

    if headstage_number == 4:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,64]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

        datatp=data.transpose()#Array from openephys has to be transposed to match RawArray MNE function to create.
        del data
        del data_adc

        'Below I make the channel names and channel types, this should go in the parameteres file later.'

        n_channels=65

        channel_names=['hpc_mid_deep', 'hpc_mid_sup', 'hpc_ros_deep', 'hpc_ros_sup', 'pfc_deep',
                       'pfc_sup', 'cx1', 'cx2', 'hpc_contra_deep', 'hpc_contra_sup',
                       'EMG1', 'EMG2', 'cb_deep', 'cb_sup', 'hp_caudal_deep', 'hpc_caudal_sup',
                       'hpc_mid_d_2', 'hpc_mid_s_2', 'hpc_ros_d_2', 'hpc_ros_s_2', 'pfc_d_2',
                       'pfc_sup_2', 'cx1_2', 'cx2_2', 'hpc_ct_d_2', 'hpc_ct_s_2',
                       'EMG1_2', 'EMG2_2', 'cb_deep_2', 'cb_sup_2', 'hp_caud_d_2', 'hpc_caud_s_2',
                       'hpc_mid_d_3', 'hpc_mid_s_3', 'hpc_ros_d_3', 'hpc_ros_s_3', 'pfc_d_3',
                       'pfc_s_3', 'cx1_3', 'cx2_3', 'hpc_c_d_3', 'hpc_c_s_3',
                       'EMG1_3', 'EMG2_3', 'cb_deep_3', 'cb_sup_3', 'hp_c_d_3', 'hpc_c_s_3',
                       'hpc_mid_d_4', 'hpc_mid_s_4', 'hpc_ros_d_4', 'hpc_ros_s_4', 'pfc_d_4',
                       'pfc_s_4', 'cx1_4', 'cx2_4', 'hpc_c_d_4', 'hpc_c_s_4',
                       'EMG1_4', 'EMG2_4', 'cb_deep_4', 'cb_sup_4', 'hp_c_d_4', 'hpc_c_s_4',

                       'Opto']
        channel_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'stim']



    if headstage_number == 3:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,48]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

        datatp=data.transpose()#Array from openephys has to be transposed to match RawArray MNE function to create.
        del data
        del data_adc

        'Below I make the channel names and channel types, this should go in the parameteres file later.'

        n_channels=49

        channel_names=['hpc_mid_deep', 'hpc_mid_sup', 'hpc_ros_deep', 'hpc_ros_sup', 'pfc_deep',
                       'pfc_sup', 'cx1', 'cx2', 'hpc_contra_deep', 'hpc_contra_sup',
                       'EMG1', 'EMG2', 'cb_deep', 'cb_sup', 'hp_caudal_deep', 'hpc_caudal_sup',
                       'hpc_mid_d_2', 'hpc_mid_s_2', 'hpc_ros_d_2', 'hpc_ros_s_2', 'pfc_d_2',
                       'pfc_sup_2', 'cx1_2', 'cx2_2', 'hpc_ct_d_2', 'hpc_ct_s_2',
                       'EMG1_2', 'EMG2_2', 'cb_deep_2', 'cb_sup_2', 'hp_caud_d_2', 'hpc_caud_s_2',
                       'hpc_mid_d_3', 'hpc_mid_s_3', 'hpc_ros_d_3', 'hpc_ros_s_3', 'pfc_d_3',
                       'pfc_s_3', 'cx1_3', 'cx2_3', 'hpc_c_d_3', 'hpc_c_s_3',
                       'EMG1_3', 'EMG2_3', 'cb_deep_3', 'cb_sup_3', 'hp_c_d_3', 'hpc_c_s_3',
                       'Opto']
        channel_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'stim']

    if headstage_number ==2:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,32]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

        datatp=data.transpose()#Array from openephys has to be transposed to match RawArray MNE function to create.
        del data
        del data_adc

        n_channels=33

        channel_names=['hpc_mid_deep', 'hpc_mid_sup', 'hpc_ros_deep', 'hpc_ros_sup', 'pfc_deep',
                       'pfc_sup', 'cx1', 'cx2', 'hpc_contra_deep', 'hpc_contra_sup',
                       'EMG1', 'EMG2', 'cb_deep', 'cb_sup', 'hp_caudal_deep', 'hpc_caudal_sup',
                       'hpc_mid_d_2', 'hpc_mid_s_2', 'hpc_ros_d_2', 'hpc_ros_s_2', 'pfc_d_2',
                       'pfc_sup_2', 'cx1_2', 'cx2_2', 'hpc_ct_d_2', 'hpc_ct_s_2',
                       'EMG1_2', 'EMG2_2', 'cb_deep_2', 'cb_sup_2', 'hp_caud_d_2', 'hpc_caud_s_2',
                       'Opto']
        channel_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'stim']

    if headstage_number ==1:

        'Add Opto Stim Channel to Data'

        data= np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
        data[:,16]=(data_adc[:,0]*300) #Multiply by 300 to have about the same scale for optogenetics.

        datatp=data.transpose()#Array from openephys has to be transposed to match RawArray MNE function to create.
        del data
        del data_adc

        n_channels=17

        channel_names=['hpc_mid_deep', 'hpc_mid_sup', 'hpc_ros_deep', 'hpc_ros_sup', 'pfc_deep',
                       'pfc_sup', 'cx1', 'cx2', 'hpc_contra_deep', 'hpc_contra_sup',
                       'EMG1', 'EMG2', 'cb_deep', 'cb_sup', 'hp_caudal_deep', 'hpc_caudal_sup',
                       'Opto']
        channel_types=['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','emg','emg','eeg','eeg','eeg','eeg',
                       'stim']


    'This creates the info that goes with the channels, which is names, sampling rate, and channel types.'
    info = mne.create_info(channel_names, prm.get_sampling_rate(), channel_types)


    'This makes the object that contains all the data and info about the channels.'
    'Computations like plotting, averaging, power spectrums can be performed on this object'

    custom_raw = mne.io.RawArray( datatp, info)
    del datatp
    return custom_raw


'The function below loads individual 32 channel probe recordings'
def load_32_EEG(foldername, montage_name, source_number):

    'Below are 2 functions from OpenEphys to load data channels and auxilary (accelerometer) channels'
    data=loadFolderToArray(foldername, channels = 'all', chprefix = 'CH', dtype = float, session = '0', source = source_number)
    data_aux=loadFolderToArray(foldername, channels = 'all', chprefix = 'AUX', dtype = float, session = '0', source = source_number)

    #data=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'CH', dtype = float, session = '0', source = '101')
    #data_aux=loadFolderToArray(prm.get_filepath(), channels = 'all', chprefix = 'AUX', dtype = float, session = '0', source = '101')

    'Below we append a line to the data array and add the accelrometer data. We transpose to fit the MNE data format.'
    data = np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
    data[:,32]=data_aux[:,0]*800

    #To work with standard MNE montages it is necessary to add 3 extra channels for the position of left and right ear and nose.
    data = np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
    data[:,33]=data[:,32]
    data = np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
    data[:,34]=data[:,32]
    data = np.append(data, (np.zeros((data.shape[0],1), dtype=int64)), axis=1)
    data[:,35]=data[:,32]

    datatp=data.transpose() #Array from openephys has to be transposed to match RawArray MNE function to create.
    del data

    if isinstance('montage_name', str):
        montage = mne.channels.read_montage(montage_name)
    else:
        print("The montage name is not valid")

    channel_types=['eeg','eeg','eeg','eeg','eeg','eeg', 'eeg', 'eeg',
                   'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                   ,'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg'
                   ,'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg', 'emg', 'emg', 'emg', 'emg']

    info = mne.create_info(montage.ch_names, prm.get_sampling_rate(), ch_types=channel_types,
                           montage=montage)

    'This makes the object that contains all the data and info about the channels.'
    'Computations like plotting, averaging, power spectrums can be performed on this object'

    custom_raw = mne.io.RawArray( datatp, info)
    del datatp
    return custom_raw


def electrode_combinations(montage_name, neighbors_dist, long_distance):
  montage = mne.channels.read_montage(montage_name)
  electrode_names = montage.ch_names[0:32]
  electrode_pos = 1000*montage.pos[0:32,0:2] # in mm

  distances_btw_electrodes = spatial.distance.pdist(electrode_pos, 'euclidean')

  nums = np.linspace(0, 31, 32, dtype = int)

  comb = combinations(nums, 2)
  # working with the combination element is difficult and it can only be assigned once -> it is transformed into a list
  comb_long_distance = list(comb)

  comb = combinations(nums, 2)
  comb_short_distance = list(comb)

  # indexes for the elements to delete
  indexes_to_delete_in_short_distance = []
  indexes_to_delete_in_long_distance = []

  nei = 0
  s_d = 0
  l_d = 0

  for i in range(len(distances_btw_electrodes)):
      if distances_btw_electrodes[i] <= neighbors_dist:
          indexes_to_delete_in_long_distance.append(i)
          indexes_to_delete_in_short_distance.append(i)
          nei += 1
      elif distances_btw_electrodes[i] <long_distance:
          indexes_to_delete_in_long_distance.append(i)
          s_d += 1
      else:
          indexes_to_delete_in_short_distance.append(i)
          l_d += 1

  long_dist_electrodes = np.delete(distances_btw_electrodes, indexes_to_delete_in_long_distance)
  short_dist_electrodes = np.delete(distances_btw_electrodes, indexes_to_delete_in_short_distance)

  # when it deletes an element, it is necessary to update the indexes substracting one to the total of them the list has.
  indexes_already_del = 0
  for i in range(len(indexes_to_delete_in_long_distance)):
      del comb_long_distance[indexes_to_delete_in_long_distance[i] - indexes_already_del]
      indexes_already_del += 1

  indexes_already_del = 0
  for i in range(len(indexes_to_delete_in_short_distance)):
      del comb_short_distance[indexes_to_delete_in_short_distance[i] - indexes_already_del]
      indexes_already_del += 1

  return comb_short_distance, comb_long_distance


def create_epochs(analysis_times, sampling_rate): #Makes epoch file for MNE of stimulation times.

    num_rows, num_cols=analysis_times.shape
    epochs= tile(0, (num_rows, 3))
    for n in range(0, num_rows):
        start_time=(analysis_times.item(n,0))
        epochs[n][0]=start_time*sampling_rate
        epochs[n][2]=n


    return epochs

def create_brain_state_epochs(analysis_times, sampling_rate): #Makes epoch file for MNE of stimulation times.

    num_rows, num_cols=analysis_times.shape
    epochs= tile(0, (num_rows, 3))
    for n in range(0, num_rows):
        start_time=(analysis_times.item(n,1))
        ident=(analysis_times.item(n,0))
        epochs[n][0]=start_time
        epochs[n][2]=ident
        epochs[n][1]=n


    return epochs

# Read the brain states of a recording from a .xls file
# Every cell is the state during a bin of 10 s
# Make sure values are not empty.
def import_brain_states(excelfile):
    print(" Openning Excelfile " + excelfile)
    book= xlrd.open_workbook(excelfile)
    sheet_names = book.sheet_names()
    sheet= book.sheet_by_name(sheet_names[0])
    brain_state = np.zeros(shape=(sheet.nrows, 1))

    for n in range(0, sheet.nrows-1):
        cell1=sheet.cell(n, 0)
        cell1_value=cell1.value
        brain_state[n, 0]=cell1_value

    return brain_state

def actual_stim_times(data, sampling_rate):##for use with normal opto

    times=[]
    times=data[:,64]>300
    start_times=[]

    for n in range(len(times)):
        if times[n] == True:
            start_times.append(n)

    stim_times=[]

    for n in range(len(start_times)):
        x=n-1
        if n ==0:
            stim_times.append(start_times[n]/sampling_rate)
        elif start_times[n]-start_times[n-1]>(sampling_rate*20):
            stim_times.append(start_times[n]/sampling_rate)


    stimulations= asarray(stim_times)

    return stimulations





def import_brain_state(excelfile): #Import analysis times for optogenetic on and control times from excel.
    #Make sure values are not empty.
    #Save file as .xls"
    print(" Openning Brain State Data " + excelfile)
    book= xlrd.open_workbook(excelfile)
    sheet_names = book.sheet_names()
    sheet= book.sheet_by_name(sheet_names[0])
    analysis_times= zeros(shape=(sheet.nrows, 3))
    print(analysis_times.shape)
    for n in range(0, sheet.nrows):
        cell1=sheet.cell(n, 0)
        cell1_value=cell1.value
        analysis_times[n, 0]=cell1_value

        cell2=sheet.cell(n, 1)
        cell2_value=cell2.value
        analysis_times[n, 1]=cell2_value

        cell3=sheet.cell(n, 2)
        cell3_value=cell3.value
        analysis_times[n, 2]=cell3_value

    return analysis_times


def import_channel_combo(excelfile):
    print(" Openning Channel Data " + excelfile)
    book= xlrd.open_workbook(excelfile)
    sheet_names = book.sheet_names()
    sheet= book.sheet_by_name(sheet_names[0])
    channel_combo= zeros(shape=(sheet.nrows, 2))

    for n in range(0, sheet.nrows):
        cell1=sheet.cell(n, 0)
        cell1_value=cell1.value
        channel_combo[n, 0]=cell1_value
        cell2=sheet.cell(n, 1)
        cell2_value=cell2.value
        channel_combo[n, 1]=cell2_value

    return channel_combo


def import_spreadsheet(excelfile): #Import analysis times for optogenetic on and control times from excel.
    #Make sure values are not empty.
    #Save file as .xls"
    print(" Openning Excelfile " + excelfile)
    book= xlrd.open_workbook(excelfile)
    sheet_names = book.sheet_names()
    sheet= book.sheet_by_name(sheet_names[0])
    analysis_times= zeros(shape=(sheet.nrows - 1, 5))

    for n in range(0, sheet.nrows-1):
        cell1=sheet.cell(n+1, 3)
        cell1_value=cell1.value
        analysis_times[n, 0]=cell1_value

        # Every cell is empty in the next columns in 300abs_thresh.xls!!
        # =============================================================================
        #         cell2=sheet.cell(n+1, 4)
        #         cell2_value=cell2.value
        #         analysis_times[n, 1]=cell2_value
        #         cell5=sheet.cell(n+1, 5)
        #         cell5_value=cell5.value
        #         analysis_times[n, 2]=cell5_value
        #         cell6=sheet.cell(n+1, 6)
        #         cell6_value=cell6.value
        #         analysis_times[n, 3]=cell6_value
        #         cell7=sheet.cell(n+1, 7)
        #         cell7_value=cell7.value
        #         analysis_times[n, 4]=cell7_value
        # =============================================================================

    return analysis_times


def sub_time_data(data, start_time, end_time, sampling_rate): #Gets time axis and data of specific times.

    prm.set_filelength(len(data))
    filelength=prm.get_filelength()
    timelength=filelength/sampling_rate
    time_axis = linspace(start_time, end_time, ((end_time-start_time)*sampling_rate))

    index_start = start_time*sampling_rate
    index_end = end_time*sampling_rate
    sub_data = data[int(index_start):int(index_end)]

    return time_axis, sub_data

# https://stackoverflow.com/a/2415343/12094741
def weighted_avg_sem(values, weights, id_rec):
    """
    Return the weighted average and standard error of the mean.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    if (variance < 0):
      print(f"For this recording: {id_rec}")
      print(f"variance was {variance}")
      print("values")
      print(np.array(values))
      print('weights')
      print(np.array(weights))
      print(f"average = {average}")
      variance = abs(np.average((values-average)**2, weights=weights))

    std_dev = math.sqrt(variance)
    sem_w = std_dev/len(values)
    return (average, sem_w)


def plot_all(data, sampling_rate, color):  #This allows for an initial plot of all the ddata.

    timemax=len(data)
    timelength = timemax/sampling_rate

    timeforplot = linspace(0, timelength, timemax)
    plt.plot(timeforplot, data, color)

    return


# Printing and exporting to excel individual coherences
def df_to_excel(filename, data_frame, sheet_n):
  """
    Write a dataframe in a particular sheet of a chosen excel file.

    If the file does not exist, it is created
    If the file already exists, it is not overwritten

    """
  file = pathlib.Path(filename)
  if file.exists ():
    book = load_workbook(filename)
    # xlsxwrite does not allow to open an excel file -> openpyxl
    writer = pd.ExcelWriter(filename, engine = 'openpyxl')
    writer.book = book
  else:
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

  data_frame.to_excel(writer, sheet_name=sheet_n)
  writer.save()
  writer.close()



