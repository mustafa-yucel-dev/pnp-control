import qontrol
import sys
import time
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import nidaqmx


# Setup Qontroller
serial_port_name = "COM5"
q = qontrol.QXOutput(serial_port_name=serial_port_name, response_timeout=0.1)
print("Qontroller '{:}' initialised with firmware {:} and {:} channels".format(q.device_id, q.firmware, q.n_chs))


def read_channel_voltage():
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai1")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai2")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai3")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai5")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai6")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai7")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai16")
    task.ai_channels.add_ai_voltage_chan("Dev1/ai17")
    data = task.read()
    task.close()
    return data

print(read_channel_voltage())

def load_pickle_files_in_folder(folder):
    dac_voltages = []
    nidaq_voltages = []

    # Get all file names in the folder
    file_names = os.listdir(folder)

    # Loop through the file names
    for file_name in file_names:
        # Check if the file is a pickle file
        if file_name.endswith(".pkl"):
            # Load pickle file
            file_path = os.path.join(folder, file_name)
            with open(file_path, "wb") as file:
                loaded_data_dict = pickle.load(file)
                dac_voltages.append(loaded_data_dict["DAC_Channels_"][38])
                nidaq_voltages.append(loaded_data_dict["NiDAQ_Channels_"][0])
    return dac_voltages, nidaq_voltages


# Specify the folder to save the pickle files
folder = "data_folder"
channel = 39

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

NiDAQ_Channels_ = [None for _ in range(10)]
DAC_Channels_ = [None for _ in range(q.n_chs)]

# Example dictionary
data = {
    "DAC_Channels_": DAC_Channels_,
    "NiDAQ_Channels_": NiDAQ_Channels_,
    "Wavelength": 1550.0000
}
q.v[39] = 0
while q.v[39] <= 10:
    print("HEY")
    data["DAC_Channels_"] = [q.v[i] for i in range(q.n_chs)]
    data["NiDAQ_Channels_"] = read_channel_voltage()

    # Specify the filename
    filename = "ch{}_{}V.pkl".format(channel, q.v[channel])
    # Create the file path by joining the folder and filename
    file_path = os.path.join(folder, filename)
    # Save dictionary using pickle in the specified folder
    with open(file_path, "wb") as file:
        pickle.dump(data, file)
    q.v[39] = q.v[39] + 1
    time.sleep(0.001)

q[39] = 0
q.close()

X, Y = load_pickle_files_in_folder(folder)
print(X)
