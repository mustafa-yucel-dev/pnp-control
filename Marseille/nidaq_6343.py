import nidaqmx # Import the package NIDAQMX


task = nidaqmx.Task() # Instantiate the NI DAQ


task.ai_channels.add_ai_voltage_chan("Dev1/ai0") # Read out and add the voltage of the channel AI0 which correspond to Out(1,0)


data = task.read() # Read the added voltages and assign to data
print(data)


task.ai_channels.add_ai_voltage_chan("Dev1/ai1") # Same for the second output


data = task.read()
print(data)


task.close() # Close the device