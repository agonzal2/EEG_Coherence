from scipy.signal import coherence, decimate, lfilter
import numpy as np
import multiprocessing as mp
import time

class parallel_coh():

  def __init__(self, voltage_state, sampling_r, frequency_r, b, a, max_amplitude):
    self.volt_state = voltage_state
    self.sampling_r = sampling_r
    self.frequency_r = frequency_r
    self.fnotch = 50 # notching filter at 50 Hz
    self.b = b
    self.a = a
    self.max_amp = max_amplitude

  def calculate(self, first_elect, sec_elect):
    first_voltages = self.volt_state[:, first_elect + 2]
    second_voltages =  self.volt_state[:, sec_elect + 2]
   
    filtered_first = lfilter(self.b, self.a, first_voltages) # filtering power notch at 50 Hz
    filtered_second = lfilter(self.b, self.a, second_voltages)
    f_loop, Cxy_loop = coherence(filtered_first, filtered_second, self.sampling_r, nperseg=self.frequency_r)
    return f_loop, Cxy_loop



class session_coherence():

  def __init__(self, raw_times, raw_voltages, brain_states, downsampling,
                montage_name, n_electrodes, sampling_rate, brain_state):
    self.raw_times = raw_times
    self.raw_data = raw_voltages
    self.brain_states = brain_states
    self.k_down = downsampling
    self.montage_name = montage_name
    self.n_electrodes = n_electrodes
    self.downsampling_rate = sampling_rate/downsampling
    self.downfreq_ratio = sampling_rate*2/downsampling
    self.f_ratio = 2 # 2 samples per Hz
    self.down_voltages = []
    self.brain_state = brain_state # 0, 1, 2, 4. 3 for 1 and 2 together.
    self.coherence_short = []
    self.coherence_long = []
    self.f_long = []
    self.f_short =[]
    self.z_long = []
    self.z_short = []
    self.volt_state = []
    self.time_wake = 0
    self.time_REM = 0
    self.time_NoREM = 0

  # returns the downsampling data for the chosen brain state (wake, rem, nrem or convulsion)
  def downsample_data(self, amp_filter=300):
    repetitions = int(5000/self.k_down)
    brain_states_ms = np.repeat(self.brain_states, repetitions)
    raw_times = self.raw_times[0::self.k_down]

    # If custom_raw.times is bigger than brain_states_ms, we truncate custom_raw.times when we create a 2d array with both
    size_brain_states = np.size(brain_states_ms)
    state_voltage_list = [brain_states_ms, raw_times[0:size_brain_states]]

    raw_data_downsampled = []
    for i in np.arange(32):
        raw_data_downsampled.append(decimate(self.raw_data[i,:], self.k_down))

    for i in np.arange(32):
        state_voltage_list.append(raw_data_downsampled[i][0:size_brain_states])

    state_voltage_array = np.transpose(np.stack(state_voltage_list))
    # removing glitches
    state_voltage_array = state_voltage_array[abs(state_voltage_array[:,0]) < amp_filter, :]
    # Wake
    self.volt_wake = state_voltage_array[state_voltage_array[:,0] == 0, :]
    self.time_wake = np.size(self.volt_wake)
    #Non-REM
    self.volt_nrem = state_voltage_array[state_voltage_array[:,0] == 1, :]
    self.time_NoREM = np.size(self.volt_nrem)
    #REM
    self.volt_rem = state_voltage_array[state_voltage_array[:,0] == 2, :]
    self.time_REM = np.size(self.volt_rem)
    #Convulsion
    self.volt_convulsion = state_voltage_array[state_voltage_array[:,0] == 4, :]
    self.time_convulsion = np.size(self.volt_convulsion)
    #Non-convulsion
    self.volt_non_convulsion = state_voltage_array[state_voltage_array[:,0] != 4, :]
    self.time_non_convulsion = np.size(self.volt_non_convulsion)
    self.raw_times = []
    del brain_states_ms
    del state_voltage_list
    del state_voltage_array
    del raw_data_downsampled
    del self.raw_times
    del raw_times
    #del volt_int
    #del volt_sleeping

  # It will be different depending on the brain state
  def calc_cohe_short(self, comb_short_distance, max_ampl = 300, brain_state=0, s_processes=34, s_chunk=1, b = np.array([0,0,0]), a=np.array([0,0,0])):
    self.choose_voltage_state(brain_state) # it defines which is going to be the current volt_state

    ### PARALLEL ###
    start_time = time.time()
    coh_short = parallel_coh(self.volt_state, self.downsampling_rate, self.downfreq_ratio, b, a, max_ampl)
    pool = mp.Pool(s_processes)
    # starmap only returns one value, even if the function returns more than one
    coherence_short_parallel = pool.starmap(coh_short.calculate, comb_short_distance, chunksize=s_chunk)
    pool.close()
    self.f_short = np.asarray(coherence_short_parallel[0][0]) # just need a frequency array. They are all the same
    coherences = []
    for t in list(coherence_short_parallel):
      coherences.append(coherence_short_parallel[0][1])
    self.coherence_short = np.asarray(coherences)
    print(f'--- The SHORT distance coherence took {(time.time() - start_time)} seconds ---')


  def calc_cohe_long(self, comb_long_distance, max_ampl = 300, brain_state=0, l_processes=48, l_chunk=1, b = np.array([0,0,0]), a=np.array([0,0,0])):

    self.choose_voltage_state(brain_state) # it defines which is going to be the current volt_state

    ### PARALLEL ###
    start_time = time.time()
    coh_long = parallel_coh(self.volt_state, self.downsampling_rate, self.downfreq_ratio, b, a, max_ampl)
    pool = mp.Pool(l_processes)
    # starmap only returns one value, even if the function returns more than one
    coherence_long_parallel = pool.starmap(coh_long.calculate, comb_long_distance, chunksize=l_chunk)
    pool.close()
    self.f_long = np.asarray(coherence_long_parallel[0][0]) # just need a frequency array. They are all the same
    coherences = []
    for t in list(coherence_long_parallel):
      coherences.append(coherence_long_parallel[0][1])
    print(f'--- The LONG distance coherence took {(time.time() - start_time)} seconds ---')
    self.coherence_long = np.asarray(coherences)


  def calc_zcoh_short(self, f_list = []):
    k_top_freq = self.set_top_freq()
    self.f_w=self.f_short[1*self.f_ratio: k_top_freq*self.f_ratio + 1] # 1.5-100 Hz
    Cxy_w_short  = self.coherence_short[:, 0*self.f_ratio: k_top_freq*self.f_ratio + 1]
    # First pass everything to z
    Cxy_w_short_z = np.arctanh(Cxy_w_short)
    # Then average in z
    # For line plotting (array of numbers, from 1.5 Hz to k_top_freq Hz bins)
    short_line_plot_m_z=np.mean(Cxy_w_short_z[:, 1*self.f_ratio : k_top_freq*self.f_ratio + 1], axis=0)
    # Z inverse transform
    self.short_line_plot_1rec_m = np.tanh(short_line_plot_m_z)

    # Same for every freq band. Both for bar plots and significance statistics
    self.short_1rec_m = []
    for freq_band in f_list:
      short_m_z = np.mean(Cxy_w_short_z[:, freq_band[1]*self.f_ratio : freq_band[2]*self.f_ratio + 1], axis=(0,1))
      self.short_1rec_m.append(np.tanh(short_m_z))    
    

  def calc_zcoh_long(self, f_list):
    k_top_freq = self.set_top_freq()
    self.f_w=self.f_long[1*self.f_ratio: k_top_freq*self.f_ratio + 1] # 1.5-k_top_freq Hz
    Cxy_w_long  = self.coherence_long[:, 0*self.f_ratio: k_top_freq*self.f_ratio + 1]
    # First pass everything to z
    Cxy_w_long_z = np.arctanh(Cxy_w_long)
    # Then average in z
    long_line_plot_m_z=np.mean(Cxy_w_long_z[:, 1*self.f_ratio : k_top_freq*self.f_ratio + 1], axis=0)
    self.long_line_plot_1rec_m = np.tanh(long_line_plot_m_z)

    # Same for every freq band. Both for bar plots and significance statistics
    self.long_1rec_m = []
    for freq_band in f_list:
      long_m_z = np.mean(Cxy_w_long_z[:, freq_band[1]*self.f_ratio : freq_band[2]*self.f_ratio + 1], axis=(0,1))
      self.long_1rec_m.append(np.tanh(long_m_z))

  def set_top_freq(self):
    if self.downsampling_rate == 1000:
      top_freq = 400
    elif self.downsampling_rate == 500:
      top_freq = 200
    elif self.downsampling_rate == 250:
      top_freq = 100
    else:
      top_freq = 50    
    return top_freq
    
    
  def choose_voltage_state(self, brain_state = 0):
    if brain_state == 0:
      self.volt_state = self.volt_wake
      self.time_state = self.time_wake
    elif brain_state == 1:
      self.volt_state = self.volt_nrem
      self.time_state = self.time_NoREM
    elif brain_state == 2:
      self.volt_state = self.volt_rem
      self.time_state = self.time_REM
    #elif brain_state == 3:
    #  self.volt_state = self.volt_sleeping
    #  self.time_state = self.time_sleeping
    elif brain_state == 4:
      self.volt_state = self.volt_convulsion
      self.time_state = self.time_convulsion
    elif brain_state == 5:
      self.volt_state = self.volt_non_convulsion
      self.time_state = self.time_non_convulsion
    else:
      print (f'Error with the brain state: {brain_state}')


  def parallel_coh(self, first_elect, sec_elect):
    f_loop, Cxy_loop = coherence(self.volt_state[:, first_elect + 2], self.volt_state[:, sec_elect + 2], self.downsampling_rate, nperseg=self.downfreq_ratio)
    return f_loop, Cxy_loop