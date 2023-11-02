import matplotlib.pyplot as plt
import pyvisa as visa
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import numpy as np
import sys
import os
import qontrol
import pickle as pkl
import scipy.signal
from santectsl570 import SantecTSL570
from datetime import datetime
import re

def channels_arrangement(nums):
    """
    Arrange a list of channel numbers into a more readable format.

    Parameters:
        nums (list of int): A list of channel numbers.

    Returns:
        list of str: A list of strings representing the arranged channel names.
    """
    if not nums:
        return
    
    result = []
    current_range = [nums[0]]
    channels_arranged = []

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            current_range.append(nums[i])
        else:
            result.append(current_range)
            current_range = [nums[i]]

    result.append(current_range)

    for r in result:
        if len(r) == 1:
            channels_arranged.append(f"Dev1/ai{r[0]}")
        else:
            channels_arranged.append(f"Dev1/ai{r[0]}:{r[-1]}")
            
    return channels_arranged


def arrange_data_V_I(V, I):
    """
    Rearranges voltage and current data from a list of measurements to separate lists for each channel.

    Parameters:
        V (list of lists): List of voltage measurements for each measurement step and channel.
        I (list of lists): List of current measurements for each measurement step and channel.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains voltage data arranged by channel.
            - The second list contains current data arranged by channel.
    """
    N_STEP = len(V)
    N_CHANNEL = len(V[0])
    V_arranged = []
    I_arranged = []

    for ch_index in range(N_CHANNEL):
        V_temp = []
        I_temp = []
        for meas_index in range(N_STEP):
            V_temp.append(V[meas_index][ch_index])
            I_temp.append(I[meas_index][ch_index])

        V_arranged.append(V_temp)
        I_arranged.append(I_temp)

    return V_arranged, I_arranged


def maxmin(powers, sweeping_value):
    '''
    Analyzes multiple power data sets with respect to the sweeping value and returns the maximum and minimum values 
    of each data set in a nested list.
    
    Parameters:
    - powers: List of power data sets (e.g., [power1, power2,..., powerN])
    - voltsweeping_valueage: The common voltage array for all power readings
    
    Returns:
    - List of [min, max] pairs for each power data set
    '''

    # List to store maximums and minimums
    max_values = []
    min_values = []
    
    # List to store x values corresponding to maximums and minimums
    x_max_values = []
    x_min_values = []
    
    colors = ["red", "blue", "green", "purple", "yellow", "cyan", "magenta", "orange", "pink", "brown"] # Add more colors if needed

    for idx, power in enumerate(powers):
        pnorm = scipy.signal.savgol_filter(power, 20, 6)
        
        # Find peaks (maxima)
        peaks, _ = scipy.signal.find_peaks(pnorm)
        
        # Find troughs (minima)
        troughs, _ = scipy.signal.find_peaks(-pnorm, prominence=-0.1)  # Use a negative prominence value

        max_power = [pnorm[p] for p in peaks]
        min_power = [pnorm[t] for t in troughs]
        
        # Corresponding X values for maxima and minima rounded to one decimal place
        x_max = [round(sweeping_value[p], 1) for p in peaks]
        x_min = [round(sweeping_value[t], 1) for t in troughs]

        # Plot the curve first
        plt.plot(sweeping_value, pnorm, color=colors[idx % len(colors)], zorder=1)

        # Plot the maxima and minima points on top of the curve using zorder=2
        plt.scatter(x_max, max_power, color='black', label='Maxima' if idx == 0 else "", zorder=2)
        plt.scatter(x_min, min_power, color='grey', label='Minima' if idx == 0 else "", zorder=2)
            
        max_values.append(max_power)
        min_values.append(min_power)
        
        x_max_values.append(x_max)
        x_min_values.append(x_min)


    plt.legend()
    plt.show()

    return x_max_values, x_min_values

def to_mean_power(V, I):
    """
    to_mean_power: Calculate the power (in watts) using the formula P = V * I.
    
    Parameters:
    - V : 2D list of voltages. Inner dimension is the channel index, and outter dimension is the measurement index.
    - I : 2D list of currents. Same dimensions as V.
    
    Returns:
    - P_W_mean : 2D list of mean power values in watts. Same dimensions as V and I.
    """
    N_STEP    = len(V[0])
    N_CHANNEL = len(V)
    P_W_mean  = []

    for ch_index in range(N_CHANNEL):
        P_temp = []  # Create a new list for each channel

        for meas_index in range(N_STEP):
            # P_temp.append(np.mean(np.multiply(V[ch_index][meas_index], I[ch_index][meas_index])))
            P_temp.append(np.mean(V[ch_index][meas_index]))
        P_W_mean.append(P_temp)


    return P_W_mean


import numpy as np

def to_mean_voltage_current(V, I):
    """
    Computes the mean voltage and current values for each channel and measurement step.

    Parameters:
        V (list of lists): List of voltage measurements for each channel and measurement step.
        I (list of lists): List of current measurements for each channel and measurement step.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains the mean voltage values for each channel and measurement step.
            - The second list contains the mean current values for each channel and measurement step.

    Example:
        If you have voltage and current data organized as lists of lists, you can use this function as follows:
        
        V_data = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        I_data = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        
        V_mean, I_mean = to_mean_voltage_current(V_data, I_data)
        # V_mean and I_mean will contain the mean voltage and current values, respectively.
    """
    N_STEP = len(V[0])
    N_CHANNEL = len(V)
    V_mean = []
    I_mean = []
    
    for ch_index in range(N_CHANNEL):
        
        V_temp = []
        I_temp = []

        for meas_index in range(N_STEP):
            V_temp.append(np.mean(V[ch_index][meas_index]))
            I_temp.append(np.mean(I[ch_index][meas_index]))

        V_mean.append(V_temp)
        I_mean.append(I_temp)

    return (V_mean, I_mean)



def read_pkls_in_current_directory(folder):
    """
    read_pkls_in_current_directory: Returns a list of dictionaries loaded from .pkl files in the specified folder.

    Parameters:
        folder (str): The name of the folder containing the .pkl files in the current directory.

    Returns:
        list: A list of dictionaries loaded from the .pkl files in the specified folder.

    Example:
        If you have a folder named 'my_data' in the current directory containing .pkl files,
        you can use this function as follows:
        
        data_list = read_pkls_in_current_directory('my_data')
        # data_list will contain dictionaries loaded from the .pkl files.
    """
    # Get the current directory
    current_directory = os.getcwd()
    dictionnary_directory = os.path.join(current_directory, folder)
    
    # Initialize an empty list to store dictionaries and their relative paths
    dicts_and_paths = []

    # Iterate over all files in the specified directory
    for filename in os.listdir(dictionnary_directory):
        # Check if the file is a .pkl file
        if filename.endswith(".pkl"):
            
            full_path = os.path.join(dictionnary_directory, filename)
            
            # Read the dictionary from the .pkl file
            with open(full_path, 'rb') as pickle_file:
                data_dict = pkl.load(pickle_file)
            
            # Append the dictionary to the list
            dicts_and_paths.append(data_dict)

    return dicts_and_paths


def power_to_dBm(power_watts):
    """
    Convert power from Watts to dBm.
    :param power_watts: List or array of power values in Watts.
    :return: List or array of power values in dBm.
    """
    power_dBm = 10 * np.log10(np.array(power_watts) * 1000)  # Convert to milliwatts then take 10*log10
    return power_dBm

    
def get_voltage(channels_list, sampling_rate, n_sample, AcqType):
    """
    get_voltage: This function acquire the voltage
    
    Parameters :
    - channels_list: This list refers to the channels for which the device acquires.
    - sampling_rate: The sampling rate of the ADC.
    - n_sample: The number of sample to be acquired.
    - AcqType: This defines the acquisiton type of the ADC. It's specific to the ADC.
    """
    with nidaqmx.Task() as task:
        # Add channels
        for ch in channels_arrangement(channels_list):
            task.ai_channels.add_ai_voltage_chan(ch, min_val=0, max_val=0.5)

        # Configure the sampling
        task.timing.cfg_samp_clk_timing(rate=sampling_rate,
                                        sample_mode=AcqType,
                                        samps_per_chan=n_sample)

        # Read out the task
        voltage_data = task.read(number_of_samples_per_channel=n_sample)

        return voltage_data

def get_current(channels_list, sampling_rate, n_sample, AcqType):
    """
    get_current: This function acquire the current
    
    Parameters :
    - channels_list: This list refers to the channels for which the device acquires.
    - sampling_rate: The sampling rate of the ADC.
    - n_sample: The number of sample to be acquired.
    - AcqType: This defines the acquisiton type of the ADC. It's specific to the ADC.
    """
    with nidaqmx.Task() as task:
        # Add channels
        for ch in channels_arrangement(channels_list):
            task.ai_channels.add_ai_current_chan(ch, min_val=0, max_val=0.01)

        # Configure the sampling
        task.timing.cfg_samp_clk_timing(rate=sampling_rate,
                                        sample_mode=AcqType,
                                        samps_per_chan=n_sample)

        # Read out the task
        current_data = task.read(number_of_samples_per_channel=n_sample)

        return current_data

def get_measurements(channels_list, sampling_rate, n_sample):
    """
    get_measurements: This function acquire the voltage and current separately
    
    Parameters :
    - channels_list: This list refers to the channels for which the device acquires.
    - sampling_rate: The sampling rate of the ADC.
    - n_sample: The number of sample to be acquired.
    """
    V = get_voltage(channels_list, sampling_rate, n_sample, AcquisitionType.CONTINUOUS)
    I = get_current(channels_list, sampling_rate, n_sample, AcquisitionType.CONTINUOUS)

    return (V, I)

def sweep_voltage_collect_ports(phase_shifter_port, 
                                channels_list, 
                                sampling_rate,
                                n_sample,
                                v_max=None, 
                                step_iv=None,
                                phase_shifter_set=None):
    """
    sweep_voltage_collect_ports: This function acquires the voltage and current of the specified output 
    ports of the NiDAQ.

    Parameters:
    - phase_shifter_port: The number associated with the phase shifter to be tuned.
    - channels_list: A list of integers related to the NIDAQ outputs to be read. For example, if the list is [1, 2, 6], 
                        the channels to be added to the Nidaq Task would be "Dev1/ai1:2" and "Dev1/ai6". This is taken 
                        care of in get_measurements.
    - sampling_rate: The sampling rate for data acquisition.
    - n_sample: The number of samples to acquire.

    - v_max: The maximum voltage to be tuned to (used for sweeping).

    - step_iv: The step size for sweeping.
    - phase_shifter_set: A list of phase shifters that represent the states of MZIs (e.g., bar or cross state). 
                            If None, all MZIs have a random phase shift.
    """
    # PhD : Photo Diode measured values
    V_PhD = [] 
    I_PhD = []
    # PhD : Phase Shifter applied and measured values
    V_PhS = []
    I_PhS = []
    
    # Connected to COM3
    serial_port_name = "COM3"
    # Setup Qontroller
    q = qontrol.QXOutput(serial_port_name=serial_port_name, response_timeout=0.1)   
    q.vmax = 12 
    # Initialize all ports to 0

    q.v[:] = 0  # Set all voltages to 0
    N_STEP = int(v_max / step_iv)

    
    if phase_shifter_set is not None:
        pass
    
    print(f"The sweep goes from {0} to {v_max} with {N_STEP} steps.")
    vvals = [float(x) for x in np.linspace(0, v_max, N_STEP)]
    for val in vvals:
            time1 = time.time()
            q.v[phase_shifter_port] = val  # Set voltage if decision is 0
            time2 = time.time()
            time.sleep(0.0001)
            print(f"Voltage {q.v[phase_shifter_port]} Current {q.i[phase_shifter_port]}")
            time3 = time.time()
            V_raw, I_raw = get_measurements(channels_list, sampling_rate, n_sample)
            
            V_PhS.append(q.v[phase_shifter_port])
            I_PhS.append(q.i[phase_shifter_port])
            time4 = time.time()
            V_PhD.append(V_raw)
            I_PhD.append(I_raw)
            
            print(f"Time to write {time2 - time1} and the time to acquire{time4-time3}")
            
    set_voltage = q.v[phase_shifter_port]
    
    print(f"The Voltage is {set_voltage} V")
    q.v[:] = 0  # Set all voltages back to 0

    print("Sweep complete.")
    
    q.close()
    
    V_PhD,I_PhD = arrange_data_V_I(V_PhD,I_PhD)
    
    return (V_PhD,I_PhD,V_PhS,I_PhS)

def sweep_current_collect_ports(phase_shifter_port, 
                                channels_list, 
                                sampling_rate,
                                n_sample,
                                i_max=None, 
                                step_iv=None,
                                phase_shifter_set=None,
                                phase_shifter_set_folder=None):
    """
    sweep_voltage_collect_ports: This function acquires the voltage and current of the specified output 
    ports of the NiDAQ.

    Parameters:
    - phase_shifter_port: The number associated with the phase shifter to be tuned.
    - channels_list: A list of integers related to the NIDAQ outputs to be read. For example, if the list is [1, 2, 6], 
                        the channels to be added to the Nidaq Task would be "Dev1/ai1:2" and "Dev1/ai6". This is taken 
                        care of in get_measurements.
    - sampling_rate: The sampling rate for data acquisition.
    - n_sample: The number of samples to acquire.

    - v_max: The maximum voltage to be tuned to (used for sweeping).

    - step_iv: The step size for sweeping.
    - phase_shifter_set: A list of phase shifters that represent the states of MZIs (e.g., bar or cross state). 
                            If None, all MZIs have a random phase shift.
    """
    # PhD : Photo Diode measured values
    V_PhD = [] 
    I_PhD = []
    # PhD : Phase Shifter applied and measured values
    V_PhS = []
    I_PhS = []
    
    # Connected to COM3
    serial_port_name = "COM3"
    # Setup Qontroller
    q = qontrol.QXOutput(serial_port_name=serial_port_name, response_timeout=0.1)   
    q.vmax = 12 

    N_STEP = int(i_max / step_iv)

    
    if phase_shifter_set is not None:
        phase_shifter_params = read_pkls_in_current_directory(phase_shifter_set_folder)
        for phase_shifter_encoder in phase_shifter_set:
            phase_shifter_i = phase_shifter_encoder[0]
            phase_shifter_state = phase_shifter_encoder[1]
            if phase_shifter_state == 0:
                for device_i_param in phase_shifter_params :
                    if device_i_param['device_name'] == phase_shifter_i:
                        current_i = float(device_i_param['min_list'][0])
                        q.i[phase_shifter_i] = current_i
                        time.sleep(0.1)
                        print(f"to be set to : {current_i}")
                        print(f"Current set to {phase_shifter_i}: {q.i[phase_shifter_i]} ")
            else:
                for device_i_param in phase_shifter_params :
                    if device_i_param['device_name'] == phase_shifter_i:
                        current_i = float(device_i_param['max_list'][0])
                        q.i[phase_shifter_i] = current_i
                        time.sleep(0.1)
                        print(f"to be set to : {current_i}")
                        print(f"Current set to {phase_shifter_i}: {q.i[phase_shifter_i]} ")
    
    print(f"The sweep goes from {0} to {i_max} with {N_STEP} steps.")
    vvals = [float(x) for x in np.linspace(0, i_max, N_STEP)]
    for val in vvals:
            time1 = time.time()
            q.i[phase_shifter_port] = val  # Set voltage if decision is 0
            time2 = time.time()
            time.sleep(0.0001)
            print(f"Voltage {q.v[phase_shifter_port]} Current {q.i[phase_shifter_port]}")
            time3 = time.time()
            
            V_raw, I_raw = get_measurements(channels_list, sampling_rate, n_sample)
            V_PhD.append(V_raw)  # Append V_raw to the list V
            I_PhD.append(I_raw)  # Append I_raw to the list I
            time4 = time.time()
            V_PhS.append(q.v[phase_shifter_port])
            I_PhS.append(q.i[phase_shifter_port])
            
            print(f"Time to write {time2 - time1} and the time to acquire{time4-time3}")
            
    set_voltage = q.i[phase_shifter_port]
    
    print(f"The Current is {set_voltage} V")
    q.v[:] = 0  # Set all voltages back to 0

    print("Sweep complete.")
    
    q.close()
    
    V_PhD,I_PhD = arrange_data_V_I(V_PhD,I_PhD)

    return (V_PhD,I_PhD,V_PhS,I_PhS)


def plot_fourier_transform(DATA, sample_rate):
    """
    TODO
    """
    colors = ["red", "blue"]
    index = 0
    for data in DATA:
        # Compute the Fast Fourier Transform (FFT) of the data
        fft_result = np.fft.fft(data)
        
        # Calculate the frequency values corresponding to the FFT result
        frequencies = np.fft.fftfreq(len(data), d=1/sample_rate)
        
        # Take the absolute value to get the magnitude (amplitude) of the complex values
        magnitude = np.abs(fft_result)
        
        # Filter frequencies and magnitudes to include only the range from -2500 to 2500
        filtered_indices = np.where((frequencies >= -1000) & (frequencies <= 1000))
        filtered_frequencies = frequencies[filtered_indices]
        filtered_magnitude = magnitude[filtered_indices]
        
        # Sort the frequencies and magnitudes based on frequencies
        sorted_indices = np.argsort(filtered_frequencies)
        sorted_frequencies = filtered_frequencies[sorted_indices]
        sorted_magnitude = filtered_magnitude[sorted_indices]

        # Plot the Fourier Transform
        plt.plot(sorted_frequencies, sorted_magnitude, color=colors[index], label="Port {}".format(index + 1))
        index += 1
    
    # Plot parameters
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def calculate_error_bars(data):
    """
    calculate_error_bars: Calculate error bars based on the maximum difference between measurements.

    Parameters:
    - data: A 2D NumPy array where each row represents measurements at the same data point, 
            and columns represent different data points.

    Returns:
    - max_diff: A 1D NumPy array containing the maximum differences between measurements at each data point.
    """
    max_diff = np.max(data, axis=0) - np.min(data, axis=0)
    return max_diff

def plot_average_with_measurements(data):
    """
    plot_average_with_measurements: Plot individual measurements with noise, the average, and return the average.

    Parameters:
    - data: A 2D NumPy array containing measurements with noise. Each row represents a measurement, 
            and columns represent different data points.

    Returns:
    - average: A 1D NumPy array containing the average of measurements at each data point.
    """
    M, N = data.shape
    x = np.arange(N)

    # Plot individual measurements
    plt.figure(figsize=(10, 8))
    for i in range(M):
        plt.plot(x, data[i], label=f'Measurement {i + 1}', marker='o', linestyle='-')
    plt.xticks(x)
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Measurements with Noise')

    # Calculate and plot the average
    average = np.mean(data, axis=0)
    error_bars = calculate_error_bars(data)

    # Adjust the size of error bars for visualization purposes
    error_bars = error_bars * 0.2  # Adjust the factor as needed for your data

    # Create the plot with filling up to the adjusted error bars
    plt.figure(figsize=(10, 8))
    plt.plot(x, average, marker='o', linestyle='-')
    plt.fill_between(x, average - error_bars, average + error_bars, alpha=0.2)  # Fill up to the adjusted error bars
    plt.xticks(x)
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Average with Error Filling (Max Difference-Based)')

    # Display the chart
    plt.show()

    return average

def plot_channels(P, values, phase_shifter_port):
    """
    plot_channels: This function plot the power of each pairs of output.
    
    Parameters :
    - P: The mean power of N channels. Its a matrix of (len(P),N_step)
    - values: List of the applied voltage or current  to the phase shifter.
    - phase_shifter_port: The port to which the sweeping is applied to.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#1c8fbd", "#f97d6d", "#bcbd22", "#17becf"]

    # Calculate the number of sub-plots needed (8 pairs)
    num_pairs = len(P) // 2

    # Create a figure with 2 rows of subplots: The first row has 4 plots, the second row has 4 plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # Set X label for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("I^2 [mA]")

    fig.suptitle("Tuning of {}".format(phase_shifter_port), fontsize=16)
    for i in range(2):  # Iterate over rows
        for j in range(4):  # Iterate over columns
            pair_index = i * 4 + j  # Calculate the index of the pair to plot
            if pair_index < num_pairs:
                pair = [pair_index * 2, pair_index * 2 + 1]
                ax = axes[i, j]
                for channel_index in pair:
                    P[channel_index] = np.array(P[channel_index])
                    # Check for negative values and adjust accordingly
                    min_val = P[channel_index].min()
                    if min_val < 0:
                        P[channel_index] += abs(min_val)
                        print("Added power to {} : {}".format(channel_index, abs(min_val)))
                    ax.plot(values, P[channel_index] , color=colors[channel_index % len(colors)],
                            label=f"Channel {channel_index}")
                    ax.set_title(f"Output pair {pair_index + 1}")
                    ax.grid(True)
                    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjusting the main title space
    plt.show()
    
def get_data(folder, get_n_meas, get_device_meas, get_channel_list):
    """
    Retrieve data from specified folders and organize it.

    This function reads data from a set of folders, performs calculations on the data, and organizes it into lists.
    
    Parameters:
    folder (str): The main folder containing subfolders with measurement data.
    get_n_meas (int): The number of measurements to retrieve from each subfolder.
    get_device_meas (list): A list of device names for which data will be retrieved.
    get_channel_list (list): A list of Nidaq channels for which data will be retrieved.

    Returns:
    V_PHD_TOT (list): A list of lists containing voltage data for specified devices and channels.
    I_PHD_TOT (list): A list of lists containing current data for specified devices and channels.
    V_PHS_TOT (list): A list of lists (initially empty) for future use.
    I_PHS_TOT (list): A list of lists (initially empty) for future use.
    V_TUNED (list): A list of lists (initially empty) for future use.
    I_TUNED (list): A list of lists (initially empty) for future use.
    """
    
    # Define the elements to be returned
    empty_list = [ [] for _ in range(len(get_device_meas)) ]
    V_PHD_TOT  = empty_list
    I_PHD_TOT  = empty_list
    V_PHS_TOT  = empty_list
    I_PHS_TOT  = empty_list
    V_TUNED    = None
    I_TUNED    = None
    DEVICES    = []

    # Verification
    n_meas = number_of_folder(folder)
    if n_meas < get_n_meas: print("error : get_n_meas > n_meas")
    
    # Loop for data
    # For N measurements in the folder
    for folder_index in range((get_n_meas)):
        meas_folder = f"{folder}\\{folder_index}"
        D = read_pkls_in_current_directory(meas_folder)
        index_device = 0
        # For every tunned devices in this measurements
        for device_i in D:
            # Get the measurements of the interest devices
            if device_i['device_name'] in get_device_meas:
                V,I = [],[]

                # Get the data of the interest nidaq channels
                for nidaq_channel in get_channel_list:
                    V.append(device_i['voltage_phd'][nidaq_channel])
                    I.append(device_i['current_phd'][nidaq_channel])
                # Average of the samples for every channels [N_Channels][N_Step][N_Sample] -> [N_Channels][N_Step][1<avg>]
                DEVICES.append(device_i['device_name'])
                V_mean = to_mean_voltage_current(V, I)[0]
                V_sum = np.array(V_mean).sum(axis=0)
                V_PHD_TOT[index_device].append(V_sum)
                index_device = index_device + 1
    
    V_TUNED = device_i['tuned_v']
    I_TUNED = device_i['tuned_i']
    if (V_TUNED is None and I_TUNED is None) or (V_TUNED is not None and I_TUNED is not None):
        print('error: tuned values error')
    else:
        if V_TUNED is None:
            X = I_TUNED
        else:
            X = V_TUNED

    return V_PHD_TOT,X,DEVICES

def number_of_folder(relative_folder_path):
    # Get the current working directory
    current_directory = os.getcwd()
    # Create the full path by joining the current directory and the relative path
    folder_path = os.path.join(current_directory, relative_folder_path)
    print (folder_path)
    num_folders = None
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List the contents of the folder
        items = os.listdir(folder_path)
        
        # Filter out only the directories
        subdirectories = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        
        # Get the number of subdirectories
        num_folders = len(subdirectories)
        
        print(f"Number of folders in {folder_path}: {num_folders}")
    else:
        print(f"Folder '{relative_folder_path}' does not exist in the current directory.")
    
    return num_folders

def get_devices_in_folder(meas_folder):
    """
    Retrieve device names from measurement data in a specific folder.

    This function reads .pkl files in a specified folder, extracts device names, and returns a sorted list of device names.

    Parameters:
    meas_folder (str): The folder containing measurement data.

    Returns:
    devices_names (list): A sorted list of device names found in the folder's measurement data.
    """
    D = read_pkls_in_current_directory(meas_folder)
    devices_names = []
    for device_i in D:
        print(device_i['device_name'])
        print(list(device_i.keys()))
        devices_names.append(device_i['device_name'])
    devices_names = sorted(devices_names)
    print(devices_names)
    return devices_names

def save_dict(data_dict=None, extend_folder=None):
    """
    save_dict: This function saves the dictionnary in a given folder.
    
    Parameters :
    - data_dict: This is the dictionnary to be saved in the folder.
    - extend_folder: This is a string that can extend the path of the folder.
    """
    device_name = data_dict["device_name"]
    wavelength = data_dict["wavelength"]

    # Define the default path to the .data folder in the current directory
    data_folder = ".data"
    
    # Extend the folder path if an extension is provided
    if extend_folder is not None:
        data_folder = os.path.join(data_folder, extend_folder)

    # Create the folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Get all existing files in the data_folder
    existing_files = os.listdir(data_folder)

    # Calculate n_measurement
    regex_pattern = fr'^{device_name}_{wavelength}_(\d+).pkl$'
    existing_measurements = [re.search(regex_pattern, f) for f in existing_files]
    n_measurement = sum(1 for match in existing_measurements if match)

    # Define the filename with date, time, device, and wavelength information
    filename = f"{device_name}_{wavelength}_{n_measurement}.pkl"

    # Define the full path to the pickle file within the .data folder
    pickle_file_path = os.path.join(data_folder, filename)

    # Save the dictionary to the pickle file
    with open(pickle_file_path, 'wb') as pickle_file:
        pkl.dump(data_dict, pickle_file)

    print(f"Data saved to {pickle_file_path}")

def save_dict_custom(data_dict=None, extend_folder=None):
    """
    save_dict: This function saves the dictionnary in a given folder.
    
    Parameters :
    - data_dict: This is the dictionnary to be saved in the folder.
    - extend_folder: This is a string that can extend the path of the folder.
    """
    device_name = data_dict["device_name"]

    # Define the default path to the .data folder in the current directory
    data_folder = ".data"
    
    # Extend the folder path if an extension is provided
    if extend_folder is not None:
        data_folder = os.path.join(data_folder, extend_folder)

    # Create the folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Get all existing files in the data_folder
    existing_files = os.listdir(data_folder)

    # Calculate n_measurement
    regex_pattern = fr'^{device_name}_(\d+).pkl$'
    existing_measurements = [re.search(regex_pattern, f) for f in existing_files]
    n_measurement = sum(1 for match in existing_measurements if match)

    # Define the filename with date, time, device, and wavelength information
    filename = f"{device_name}_{n_measurement}.pkl"

    # Define the full path to the pickle file within the .data folder
    pickle_file_path = os.path.join(data_folder, filename)

    # Save the dictionary to the pickle file
    with open(pickle_file_path, 'wb') as pickle_file:
        pkl.dump(data_dict, pickle_file)

    print(f"Data saved to {pickle_file_path}")

def save_fitting_params(data_dict, new_folder_name, file_termination, folder_path=''):
    """
    save_fitting_params: This function saves the dictionary in a given folder as a pickle file.
    
    Parameters:
    - data_dict: The dictionary to be saved in the folder.
    - folder_path: The path to the folder where the data will be saved (default is current directory).
    """
    device_name = data_dict["device_name"]


    folder_name = new_folder_name

    
    # Extend the folder path if provided
    if folder_path:
        folder_name = os.path.join(folder_path, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the filename with device and wavelength information
    filename = f"{device_name}_{file_termination}.pkl"

    # Define the full path to the pickle file
    file_path = os.path.join(folder_name, filename)

    # Save the dictionary to the pickle file
    with open(file_path, 'wb') as pickle_file:
        pkl.dump(data_dict, pickle_file)

    print(f"Data saved to {file_path}")

def device_tunning_info(device_name = None, 
                        voltage_phd = None, 
                        current_phd = None,
                        voltage_phs = None,
                        current_phs = None,
                        tuned_v     = None,
                        tuned_i     = None,
                        comment     = None):
    """
    device_tunning_info: Returns a dictionnary with relevant informations of the device.
    
    Parameters :
    - device_name: Name of the device which is tuned.
    - voltage: List of the raw acquire voltage data.
    - current: List of the raw acquire current data.
    - tuned_v: List of the tuned voltage.
    - tuned_i: List of the tuned current.
    - comment: Comment on the measurements.
    """
    # Get the wavelength from the laser
    laser = SantecTSL570("GPIB0::19::INSTR")
    wavelength = round(laser.get_wavelength()*1e9,3)
    
    # Get the current date and time
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    device_info = {
        "device_name" : device_name,
        "data_time"   : date_time,
        "voltage_phd" : voltage_phd,
        "current_phd" : current_phd,
        "voltage_phs" : voltage_phs,
        "current_phs" : current_phs,
        "tuned_v"     : tuned_v,
        "tuned_i"     : tuned_i,
        "wavelength"  : wavelength,
        "comment"     : comment
    }
    return device_info

def create_parameter_dictionary(device_name, offset, amplitude, alpha, p1, p2, p3, p4, P_pi, phase, best_r_squared):
    parameter_dict = {
        'device_name'   : device_name,
        'offset'        : offset,
        'amplitude'     : amplitude,
        'alpha'         : alpha,
        'p1'            : p1,
        'p2'            : p2,
        'p3'            : p3,
        'p4'            : p4,
        'P_pi'          : P_pi,
        'phase'         : phase,
        'Best R-squared': best_r_squared
    }
    return parameter_dict

def create_device_minmiax_info(device_name, max_list, min_list):
    device_info = {
        'device_name': device_name,
        'max_list'   : max_list,
        'min_list'   : min_list,
    }
    return device_info

def get_parameters_as_list(input_dict):
    """
    Extracts the values from a dictionary and returns them as a list.

    Parameters:
        input_dict (dict): The input dictionary containing parameter-value pairs.

    Returns:
        list: A list containing the values from the input dictionary in the same order.

    Example:
        >>> input_dict = {
        ...     'device_name': 23,
        ...     'offset': 0.06347067291376776,
        ...     'amplitude': 0.01668052352193857,
        ...     'alpha': 0.025049699738242188,
        ... }
        >>> get_parameters_as_list(input_dict)
        [23, 0.06347067291376776, 0.01668052352193857, 0.025049699738242188]
    """
    parameter_list = [value for value in input_dict.values()]
    return parameter_list


def mzi1_unitary(theta, phi, alpha, beta):
    """
    This represents the theoretical function of an MZI with two beam splitters
    and two phase shifters
    ---Phi--\       /--Theta--\      /----
    .        =alpha=           =beta=
    --------/       \---------/      \----
    """
    a = np.exp( 1j*phi )*( np.cos( alpha - beta )*np.sin( theta/2 ) + 1j*np.sin(alpha + beta)*np.cos( theta/2 ) )
    b =                    np.cos( alpha + beta )*np.cos( theta/2 ) + 1j*np.sin(alpha - beta)*np.sin( theta/2 )
    c = np.exp( 1j*phi )*( np.cos( alpha + beta )*np.cos( theta/2 ) - 1j*np.sin(alpha - beta)*np.sin( theta/2 ) )
    d =                  - np.cos( alpha - beta )*np.sin( theta/2 ) + 1j*np.sin(alpha + beta)*np.cos( theta/2 )
    return 1j*np.exp(1j*theta/2)*(np.array([[a, b], [c, d]]))


# Define the voltage_non_ohmic function with Y, Z, alpha, and beta parameters
def voltage_non_ohmic(I, p1, p2, p3, p4):
    return (p4 * I**4 + p3 * I**3 + p2 * I**2 + p1 * I)

def cross_talk(I, alpha):
    return np.exp(alpha * I )

def mzm_transfer(I, offset, amplitude, alpha, p1, p2, p3, p4, P_pi, phase):
    return offset + amplitude * cross_talk(I, alpha) * np.cos((I * voltage_non_ohmic(I, p1, p2, p3, p4)) / P_pi * np.pi + phase)