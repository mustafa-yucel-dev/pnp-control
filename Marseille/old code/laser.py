import pyvisa
import time
class Laser:
    def __init__(self, address):
        self.rm = pyvisa.ResourceManager()
        self.laser = self.rm.open_resource(address)
        print(self.rm.list_resources())
        self.turn_on_low_noise_mode(self, 1)

        
    def set_wavelength(self, wavelength):
        command = f"SETL1W:{wavelength:.4f}"
        self.laser.write(command)
        response = self.laser.read()
        if response == "Successful\r\n":
            print("Wavelength set successfully.")
        else:
            print("Error setting wavelength.")
            
    def set_power(self, power):
        command = f"SETL1P:+{power:.2f}"
        self.laser.query(command)
        response = self.laser.read()
        if response == "Successful\r\n":
            print("Power set successfully.")
        else:
            print("Error setting power.")
            
    def read_unit_info(self):
        command = "READ"
        self.laser.query(command)
        response = self.laser.read()
        print(response)
            
    def read_channel_status(self, channel):
        command = f"READL{channel}"
        self.laser.query(command)
        response = self.laser.read()
        print(response)
            
    def turn_on_ld(self, channel):
        command = f"SETL{channel}L:ON"
        self.laser.query(command)
        response = self.laser.read()
        if response == "Successful\r\n":
            print(f"LD in channel {channel} turned on.")
        else:
            print(f"Error turning on LD in channel {channel}.")
            
    def turn_off_ld(self, channel):
        command = f"SETL{channel}L:OFF"
        self.laser.query(command)
        response = self.laser.read()
        if response == "Successful\r\n":
            print(f"LD in channel {channel} turned off.")
        else:
            print(f"Error turning off LD in channel {channel}.")
        
    def turn_on_low_noise_mode(self, channel):
        command = f"SETL{channel}N:ON"
        self.laser.query(command)
        response = self.laser.read()
        if response == "Successful\r\n":
            print(f"Low noise mode turned on for channel {channel}.")
        else:
            print(f"Error turning on low noise mode for channel {channel}.")
            
    def turn_off_low_noise_mode(self, channel):
        command = f"SETL{channel}N:OFF"
        self.laser.query(command)
        response = self.laser.read()
        if response == "Successful\r\n":
            print(f"Low noise mode turned off for channel {channel}.")
        else:
            print(f"Error turning off low noise mode for channel {channel}.")



rm = pyvisa.ResourceManager()
laser = rm.open_resource("ASRL6::INSTR")  # Replace with the appropriate address
wavelength = 1530.0000  # Specify the desired wavelength
command = f"SETL1W:{wavelength:.4f}"
response = laser.query(command)
command = f"SETL1N:ON"
print("ON noise")
laser.query(command)
time.sleep(2)
command = f"SETL1L:ON"
print("ON laser")
print(laser.query(command))

power = 10  # Specify the desired power
command = f"SETL1P:+{power:.2f}"
response = laser.query(command)
print(response)

laser.close()


""" 
# Usage example
laser = Laser("ASRL6::INSTR")  # Replace with the appropriate address
laser.set_wavelength(1535.1234)  # Set the desired wavelength
laser.set_power(12.34)  # Set the desired power
laser.read_unit_info()  # Read unit manufacturer information
laser.read_channel_status(1)  # Read status information for channel 1
laser.turn_on_ld(1)  # Turn on LD in channel 1
laser.turn_off_ld(1)  # Turn off LD in channel 1
laser.turn_on_low_noise_mode(1)  # Turn on low noise mode for channel 1
laser.turn_off_low_noise_mode(1)  # Turn off low noise mode for channel 1
laser.laser.close() """