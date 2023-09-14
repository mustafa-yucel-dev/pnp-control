###########################################
#
#  API for interfacing firmware > 2.0
#
#  (c) 2019 Qontrol Systems LLP
#
###########################################


import serial, re, time
from collections import deque as fifo
from random import shuffle
from serial.tools import list_ports
import sys


Q8x_ERRORS = {0:'Unknown error.',
	1:'Over-voltage error on channel {ch}.',
	2:'Over-current error on channel {ch}.',
	3:'Power error.',
	4:'Calibration error.',
	5:'Output error.',
	10:'Unrecognised command.',
	11:'Unrecognised input parameter.',
	12:'Unrecognised channel, {ch}.',
	13:'Operation forbidden.',
	14:'Serial buffer overflow.',
	15:'Serial communication error.',
	16:'Command timed out.',
	17:'SPI error.',
	18:'ADC error.',
	19:'I2C error.',
	30:'Firmware error.',
	90:'Powered up.'}



RESPONSE_OK = 'OK\n'
ERROR_FORMAT = '[A-Za-z]{1,3}(\d+):(\d+)'


class Qontroller(object):
	"""
	Super class which handles serial communication, device identification, and logging.

		device_id = None					Device ID
		serial_port = None					Serial port object
		serial_port_name = None				Name of serial port, eg 'COM1' or '/dev/tty1'
		error_desc_dict = Q8x_ERRORS			Error code descriptions
		log = fifo(maxlen = 256)			Log FIFO of sent commands and received errors
		log_handler = None					Function which catches log dictionaries
		log_to_stdout = True				Copy new log entries to stdout
		response_timeout = 0.050			Timeout for response or error to commands
		inter_response_timeout = 0.020		Timeout for response or error to get commands

	Log handler:
	The log handler may be used to catch and dynamically handle certain errors, as they arise. In the following example, it is set up to raise a RuntimeError upon reception of errors E01, E02, and E03:

		q = Qontroller()

		fatal_errors = [1, 2, 3]

		def my_log_handler(err_dict):
			if err_dict['type'] is 'err' and err_dict['id'] in fatal_errors:
				raise RuntimeError('Caught Qontrol error "{1}" at {0} ms'.format(1000*err_dict['proctime'], err_dict['desc']))

		q.log_handler = my_log_handler

	"""


	def __init__(self, *args, **kwargs):
		"""
		Initialiser.
		"""

		# Defaults

		self.device_id = None						# Device ID (i.e. [device type]-[device number])
		self.serial_port = None						# Serial port object
		self.serial_port_name = None					# Name of serial port, eg 'COM1' or '/dev/tty1'
		self.baudrate = 115200						# Serial port baud rate (signalling frequency, Hz)
		self.error_desc_dict = Q8x_ERRORS				# Error code descriptions
		self.log = fifo(maxlen = 512)				# Log FIFO of sent commands and received errors
		self.log_handler = None						# Function which catches log dictionaries

		self.log_to_stdout = False					# Copy new log entries to stdout
		self.response_timeout = 0.050				# Timeout for RESPONSE_OK or error to set commands
		self.inter_response_timeout = 0.020			# Timeout between received messages


		# Setup Rx and Tx logs
		self.total_rx_str = ''
		self.total_tx_str = ''

		# Set a time benchmark
		self.init_time = time.time()

		# Get arguments from init

		# Populate parameters, if provided
		for para in ['device_id', 'serial_port_name', 'error_desc_dict', 'log_handler', 'log_to_stdout', 'response_timeout', 'inter_response_timeout', 'baudrate']:
			try:
				self.__setattr__(para, kwargs[para])
			except KeyError:
				continue

		# Find serial port by asking it for its device id
		if 'device_id' in kwargs:
			# Search for port with matching device ID
			ob = re.match('(Q\w+)-([0-9a-fA-F\*]+)', self.device_id)
			targ_dev_type,targ_dev_num = ob.groups()
			if ob is None:
				raise AttributeException('Entered device ID ({0}) must be of form "[device type]-[device number]" where [device number] can be hexadecimal'.format(self.device_id))

			# Find serial port based on provided device ID (randomise their order)
			candidates = []
			possible_ports = list(list_ports.comports())
			shuffle(possible_ports)
			tries = 0
			for port in possible_ports:
				for i in range(60):
					sys.stdout.write(' ')
				sys.stdout.write('\r')
				sys.stdout.write('Querying port {:}... '.format(port.device))
				sys.stdout.flush()

				try:
					# Instantiate the serial port
					self.serial_port = serial.Serial(port.device, self.baudrate, timeout=0.5)
					self.serial_port.close()
					self.serial_port.open()
					# Clear buffer
					self.serial_port.reset_input_buffer()
					self.serial_port.reset_output_buffer()
					# Transmit our challenge string
					self.serial_port.write("id?\n".encode('ascii'))
					# Receive response
					response = self.serial_port.read(size=64).decode("ascii")
					# Check if we received a response
					if response == '':
						sys.stdout.write('No response\n')
						continue
					# Match the device ID
					ob = re.match('.*((?:'+ERROR_FORMAT+')|(?:Q\w+-[0-9a-fA-F\*]+)).*', response)
					if ob is not None:
						ob = re.match('(Q\w+)-([0-9a-fA-F\*]+)\n', response)
						if ob is not None:
							sys.stdout.write('{:}\n'.format(response))
							sys.stdout.flush()
							dev_type,dev_num = ob.groups()
							candidates.append({'dev_type':dev_type, 'dev_num':dev_num, 'port':port.device})
							if dev_type == targ_dev_type and dev_num == targ_dev_num:
								self.serial_port_name = port.device
								break
						else:
							ob = re.match(ERROR_FORMAT, response)
							if ob is not None:
								sys.stdout.write('Error')
								# Try this port again later
								if tries < 3:
									sys.stdout.write('. Will try again...')
									possible_ports.append(port)
									tries += 1
								else:
									sys.stdout.write('\n')
								sys.stdout.flush()
					else:
						sys.stdout.write('Not a valid device\n'.format(response))
						sys.stdout.flush()


					# Close port
					self.serial_port.close()

				except serial.serialutil.SerialException:
					sys.stdout.write('Busy\n')
					sys.stdout.flush()
					continue

			# If the target device is not found
			if not self.serial_port.is_open:
				# Check whether we found another possibility
				for candidate in candidates:
					if candidate['dev_type'] == targ_dev_type:
						self.device_id = candidate['dev_type']+'-'+candidate['dev_num']
						self.serial_port_name = candidate['port']
						print ('Qontroller.__init__: Warning: Specified device ID ({0}) could not be found. Using device with matching type ({2}) on port {1}.'.format(kwargs['device_id'], self.serial_port_name, self.device_id))
						break
				# If no similar device exists, abort
				if all([candidate['dev_type'] != targ_dev_type for candidate in candidates]):
					raise AttributeError('Specified device ID ({0}) could not be found.'.format(kwargs['device_id']))

			print ('Using serial port {0}'.format(self.serial_port_name))
			# If serial_port_name was also specified, check that it matches the one we found.
			if ('serial_port_name' in kwargs) and (self.serial_port_name != kwargs['serial_port_name']):
				print ('Qontroller.__init__: Warning: Specified serial port ({0}) does not match the one found based on the specified device ID ({1}, {2}). Using serial port {2}.'.format(kwargs['serial_port_name'], self.device_id, self.serial_port_name))

		# Open serial port directly, get device id
		elif 'serial_port_name' in kwargs:
			# Open serial communication
			# This will throw a serial.serialutil.SerialException if busy
			self.serial_port = serial.Serial(self.serial_port_name, self.baudrate, timeout = self.response_timeout)

			# Get device ID
			# Transmit our challenge string
			# This repeated try mechanism accounts for serial ports with starting hiccups
			timed_out = True
			for t in range(3):
				# Clear buffer
				self.serial_port.reset_input_buffer()
				self.serial_port.reset_output_buffer()
				# Send challenge
				self.serial_port.write('id?\n'.encode('ascii'))
				# Receive response
				start_time = time.time()
				# Wait for first byte to arrive
				while (self.serial_port.in_waiting == 0) and (time.time() - start_time < 0.2):
					pass
				# Read response, ignoring unparsable characters
				try:
					response = self.serial_port.read(size=64).decode('ascii')
				except UnicodeDecodeError:
					response = ""
				# Parse it
				ob = re.match('.*((?:'+ERROR_FORMAT+')|(?:Q\w+-[0-9a-fA-F\*]+)).*', response)
				# Check whether it's valid
				if ob is not None:
					# Flag that we have broken out correctly
					timed_out = False
					break

			# Store the parsed value
			if not timed_out:
				self.device_id = ob.groups()[0]
				# Check if it was an error, in which case clear the stored value but proceed
				ob = re.match('((?:'+ERROR_FORMAT+')|(?:Q\w+-\*+))', self.device_id)
				if ob is not None:
					# It was an error (no ID assigned yet)
					self.device_id = None
			else:
				raise RuntimeError('Qontroller.__init__: Error: Unable to communicate with device on port {0} (received response {1}, "{2}").'.format(self.serial_port_name, ":".join("{:02x}".format(ord(c)) for c in response), response.replace('\n', '\\n')))
		else:
			raise AttributeError('At least one of serial_port_name or device_id must be specified on Qontroller initialisation. Available serial ports are:\n  serial_port_name = {:}'.format('\n  serial_port_name = '.join([port.device for port in list(list_ports.comports())])))


		# Establish contents of daisy chain
		try:
			# Ask for number of upstream devices, parse it
			try:
				chain = self.issue_command('nupall', operator = '?', target_errors = [0,10,11,12,13,14,15,16], output_regex = '(?:([^:\s]+)\s*:\s*(\d+)\n*)*')
			except:
				chain = self.issue_command('nup', operator = '?', target_errors = [0,10,11,12,13,14,15,16], output_regex = '(?:([^:\s]+)\s*:\s*(\d+)\n*)*')
			# Further parse each found device into a dictionary
			for i in range(len(chain)):
				ob = re.match('([^-]+)-([0-9a-fA-F\*]+)', chain[i][0])

				device_id = chain[i][0]
				device_type = ob.groups()[0]
				device_serial = ob.groups()[1]

				try:
					index = int(chain[i][1])
				except ValueError:
					index = -1
					print ('Qontroller.__init__: Warning: Unable to determine daisy chain index of device with ID {:}.'.format(device_id))

				# Scan out number of channels from device type
				ob = re.match('[^\d]+(\d*)[^\d]*', device_type)


				try:
					n_chs = int(ob.groups()[0])
				except ValueError:
					n_chs = -1
					print ('Qontroller.__init__: Warning: Unable to determine number of channels of device at daisy chain index {:}.'.format(index))

				chain[i] = {
					'device_id':device_id,
					'device_type':device_type,
					'device_serial':device_serial,
					'n_chs':n_chs,
					'index':index}
		except:
			chain = []
			print ('Qontroller.__init__: Warning: Unable to determine daisy chain configuration.')

		self.chain = chain


	def __del__(self):
		"""
		Destructor.
		"""
		self.close()


	def close(self):
		"""
		Release resources
		"""
		if self.serial_port is not None and self.serial_port.is_open:
			# Close serial port
			self.serial_port.close()


	def transmit (self, command_string):
		"""
		Low-level transmit data method.
		"""
		# Ensure serial port is open
		if not self.serial_port.is_open:
			self.serial_port.open()

		# Write to port
		self.serial_port.write(command_string.encode('ascii'))

		# Log it
		self.total_tx_str += command_string


	def receive (self):
		"""
		Low-level receive data method which also checks for errors.
		"""
		# Ensure serial port is open
		if not self.serial_port.is_open:
			self.serial_port.open()

		# Read from port
		lines = []
		errs = []

		# Check if there's anything in the input buffer
		while self.serial_port.in_waiting > 0:
			# Get a line from the receive buffer
			line = str(self.serial_port.readline().decode('ascii'))

			# Log it
			self.total_rx_str += line
			self.total_rx_str += '\n'

			# Check if it's an error by parsing it
			err = self.parse_error(line)
			if err is None:
				# No error, keep the line
				lines.append(line)
			else:
				# Line represents an error, add to list
				errs.append(err)

		# Add any errors we found to our log
		for err in errs:
			self.log_append(type='err', id=err['id'], ch=err['ch'], desc=err['desc'], raw=err['raw'])

		return (lines, errs)


	def log_append (self, type='err', id='', ch=0, value=0, desc='', raw=''):
		"""
		Append an event to the log, adding both a calendar- and a process-timestamp."
		"""
		# Append to log fifo
		self.log.append({'timestamp':time.asctime(), 'proctime':round(time.time()-self.init_time,3), 'type':type, 'id':id, 'ch':ch, 'value':value, 'desc':desc, 'raw':raw})
		# Send to handler function (if defined)
		if self.log_handler is not None:
			self.log_handler(self.log[-1])
		# Send to stdout (if requested)
		if self.log_to_stdout:
			self.print_log (n = 1)


	def print_log (self, n = None):
		"""
		Print the n last log entries. If n == None, print all log entries.
		"""
		if n is None:
			n = len(self.log)

		for i in range(-n,0):
			print('@ {0: 8.1f} ms, {1} : {2}'.format(1000*self.log[i]['proctime'], self.log[i]['type'], self.log[i]['desc']) )


	def parse_error (self, error_str):
		"""
		Parse an encoded error (e.g. E02:07) into its code, channel, and human-readable description.
		"""
		# Regex out the error and channel indices from the string
		ob = re.match(ERROR_FORMAT, error_str)

		# If error_str doesn't match an error, return None
		if ob is None:
			return None

		# Extract the two matched groups (i.e. the error and channel indices)
		errno,chno = ob.groups()
		errno = int(errno)
		chno = int(chno)

		# Get the error description; if none is defined, mark as unrecognised
		errdesc = self.error_desc_dict.get(errno, 'Unrecognised error code.').format(ch=chno)

		return {'type':'err', 'id':errno, 'ch':chno, 'desc':errdesc, 'raw':error_str}


	def wait (self, seconds=0.0):
		"""
		Do nothing while watching for errors on the serial bus.
		"""
		start_time = time.time()
		while time.time() < start_time + seconds:
			self.receive()


	def issue_command (self, command_id, ch=None, operator='', value=None, n_lines_requested=2**31, target_errors=None, output_regex='(.*)', special_timeout = None):
		"""
		Transmit command ([command_id][ch][operator][value]) to device, collect response.

			command_id			Command header (e.g. 'v' in 'v7=1.0')
			ch					Channel index to apply command to (e.g. '7' in 'v7=1.0')
			operator			Type of command in {?, =} (e.g. '=' in 'v7=1.0')
			value				Value of set command (e.g. '1.0' in 'v7=1.0')
			n_lines_requested	Lines of data (not error) to stop after receiving, or timeout
			target_errors		Error numbers which will be raised as RuntimeError
			special_timeout		Timeout to use for this command only (!= self.response_timeout)
		"""
		# Check for previous errors
		lines,errs = self.receive()

		# Transmit command
		if ch is None:
			ch = ''
		if value is None:
			value = ''
		tx_str = '{0}{1}{2}{3}'.format(command_id, ch, operator, value)
		self.transmit(tx_str+'\n')

		# Log it
		self.log_append(type= 'set' if operator == '=' else 'get', value=value, id=command_id, ch=ch, desc='Command: "'+tx_str+'".')

		# Receive response
		lines = []
		errs = []
		if target_errors is None:
			target_errors = []
		start_time = time.time()
		last_message_time = start_time

		while (True):

				# Break conditions
				if (RESPONSE_OK in lines):
					break
				elif (len(lines) >= n_lines_requested):
					break
				elif not all([err['id'] not in target_errors for err in errs]):
					break
				elif (time.time() - start_time > self.response_timeout):
					if (time.time() - last_message_time > self.inter_response_timeout):
						break

				# Receive data
				rec_lines,rec_errs = self.receive()

				# Update the last time a message was received
				# We won't proceed now until self.inter_response_timeout has elapsed
				if len(rec_lines) + len(rec_errs) > 0:
					last_message_time = time.time()

				# Integrate received lines and errors
				lines.extend(rec_lines)
				errs.extend(rec_errs)

				# Check whether we have received a serial comms error (E15)
				if any([err['id'] == 15 for err in errs]):
					# If we did, we should try issuing the command again, recursively
					return self.issue_command (command_id, ch, operator, value, n_lines_requested, target_errors, output_regex)

				# Check whether we have received a fatal error
				if any([err['id'] in target_errors for err in errs]):
					raise RuntimeError('Received targetted error code {0}, "{1}". Log is: \n{2}.'.format(err['id'], err['desc'], self.log))

		# We timed out.
		if len(lines) == 0 and len(errs) == 0:
			if operator == '?':
				# If we are looking for a return value, raise an error
				raise RuntimeError ('Response to command {0} timed out after {1} ms.'.format(tx_str, 1000*(last_message_time - start_time)))
			else:
				# If we are setting something, just warn the user
				print('Qontroller.set_command: Warning: Response to command {0} timed out after {1:.3f} ms.'.format(tx_str, 1000*(last_message_time - start_time)))

		# Parse the output
		values = []
		for line in lines:
			op = re.match(output_regex, line)
			if op is None:
				value = []
			else:
				value = op.groups()
			values.append(value)


		self.log_append(type= 'rcv', value=None, id=command_id, ch=ch, desc='Received: "'+str(values)+'".')

		return values


	def __getattr__(self, attr):
		"""
		Allow convenience attribute access for certain parameters
		"""
		if (attr in ['firmware', 'vfull', 'ifull', 'lifetime']):
			return self.issue_command (command_id=attr, ch=None, operator='?', n_lines_requested=1)[0][0]
		else:
			return object.__getattr__(self, attr)



class ChannelVector(object):
	"""
	Custom list class which has a fixed length but mutable (typed) elements, and which phones home when its elements are read or modified.
	"""

	set_handle = None
	get_handle = None
	valid_types = (int,float)

	def __init__(self, base_list):
		self.list = base_list


	def __len__(self):
		return len(self.list)


	def __getitem__(self, key):
		if isinstance(key, slice):
			# Handle slice key
			return [self[k] for k in range(len(self))[key.start:key.stop:key.step]]
		else:
			# Handle normal key
			if self.get_handle is not None:
				get_val = self.get_handle (key, self.list[key])
				if get_val is not None:
					self.list[key] = get_val
			return self.list[key]


	def __setitem__(self, key, value):
		if all([type(value) != valid_type for valid_type in self.valid_types]):
			raise TypeError('Attempt to set value to type {0} is forbidden. Valid types are {1}.'.format(type(value), self.valid_types))
		if isinstance(key, slice):
			# Handle slice key
			for k in range(len(self))[key.start:key.stop:key.step]:
				self[k] = value
		else:
			# Handle normal key
			if self.set_handle is not None:
				self.set_handle (key, value)
			self.list[key] = value


	def __iter__(self):
		return iter(self.list)


	def __repr__(self):
		return repr([self[i] for i in range(len(self))])



class QXOutput(Qontroller):


	def __init__(self, *args, **kwargs):
		super(QXOutput, self).__init__(*args, **kwargs)
		# super(type(self), self).__init__(*args, **kwargs)

		self.n_chs = 0
		self.v_full = 0
		self.v = None			# Channel voltages (direct access)
		self.i = None			# Channel currents (direct access)
		self.vmax = None		# Channel voltages (direct access)
		self.imax = None		# Channel currents (direct access)
		self.vmax_val = 12
		self.imax_val  = 10.0

		# Get our full-scale voltage (VFULL)
		try:
			self.v_full = float(self.issue_command('vfull', operator = '?', n_lines_requested = 1, output_regex='(?:\+|-|)([\d\.]+) V')[0][0])
		except:
			raise RuntimeError("Unable to obtain VFULL from qontroller on port {:}.".format(self.serial_port_name))

		# Get our number of channels
		try:
			# See if its in the list of kwargs
			self.n_chs = kwargs['n_chs']
			if self.n_chs <= 0 or self.n_chs == None:
				raise KeyError()
		except KeyError:
			# If not in kwargs, try to get it from the chain
			try:
				self.n_chs = sum([device['n_chs'] for device in self.chain])
			except KeyError:
				# If not, just ask the top device how many ports its got
				try:
					self.n_chs = int(self.issue_command('nchan', operator = '?', n_lines_requested = 1, target_errors = [10], output_regex = '(\d+)\n')[0][0])
				except:
					# If not, just take some random value
					self.n_chs = 8
					print ("QXOutput.__init__: Warning: Failed to obtain number of daisy-chained channels automatically. Include this as n_chs argument on initialisation to workaround.")



		# Set up output direct access
		# These initialise themselves when they are first used (i.e. the 0 init is OK)

		# Voltage
		self.v = ChannelVector([0] * self.n_chs)
		self.v.set_handle = lambda ch,val: self.set_value(ch,'v',val)
		self.v.get_handle = lambda ch,val: self.get_value(ch,'v')

		self.vmax = ChannelVector([0] * self.n_chs)
		self.vmax.set_handle = lambda ch,val: self.set_value(ch,'vmax',val)
		self.vmax.get_handle = lambda ch,val: self.get_value(ch,'vmax')

		# Current
		self.i = ChannelVector([0] * self.n_chs)
		self.i.set_handle = lambda ch,val: self.set_value(ch,'i',val)
		self.i.get_handle = lambda ch,val: self.get_value(ch,'i')

		self.imax = ChannelVector([0] * self.n_chs)
		self.imax.set_handle = lambda ch,val: self.set_value(ch,'imax',val)
		self.imax.get_handle = lambda ch,val: self.get_value(ch,'imax')

		self.initialised = True

		# set vmax and imax thresholds! check to make sure this works
		self.set_thresholds()


	def set_value (self, ch, para='v', new=0):
		self.issue_command(para, ch=ch, operator='=', value=new)

	def get_value (self, ch, para='v'):
		result = self.issue_command(para, ch = ch, operator = '?', n_lines_requested = 1, output_regex = '((?:\+|-){0,1}[\d\.]+)')
		if len(result) > 0:
			if len(result[0]) > 0:
				return float(result[0][0])
		return None

	def get_all_values (self, para='v'):
		result = self.issue_command(para+'all', operator = '?', n_lines_requested = self.n_chs, output_regex = '(?:\+|-|)([\d\.]+)', special_timeout = 2*self.response_timeout)
		if len(result) > 0:
			if len(result[0]) > 0:
				result = [float(result[i][0]) for i in range(len(result))]
				return result
		return None

	def __setattr__(self, attr, val):
		# Prevent overwrite of internal variables
		try:
			if (self.initialised is True and attr in ['v', 'i', 'vmax', 'imax', 'v_full', 'n_chs']):
				print ("QXOutput.__setattr__: Warning: Overwriting of '{:}' is forbidden.".format(attr) )
				return
		except AttributeError:
			# If we are still initialising, carry on setting variable
			pass

		object.__setattr__(self, attr, val)

	def dict_set(self, dict):
		"""takes as input a dictionary = {channel: (value, param)} and sets
		each channel to the corresponging (value, param)"""
		for channel in list(dict.keys()):
			(value, param) = dict[channel]
			if param == 'i':
				self.i[channel] = value
			if param == 'v':
				self.v[channel] = value

	def set_thresholds(self):
		"quickly set vmax or imax for all channels from the global values"
		for k in range(self.n_chs):
			self.vmax[k] = self.vmax_val
			self.imax[k] = self.imax_val
		print('Driver thresholds set @ {} V , {} mA'.format(self.vmax_val, self.imax_val))
		return None
	
	def reset_voltages(self):
		N = self.n_chs
		for voltage_i in len(N):
			self.v[voltage_i]=0
		print("Channels reset to 0.")
		return None
	
	def set_channels(self,val):
		return None


if __name__ == '__main__':
	# find serial ports by ls /dev/tty.*
	# q_serial_port='/dev/tty.usbserial-FT2MCRYP'
	q_serial_port = "COM5"
	q = QXOutput(serial_port_name=q_serial_port, response_timeout=0.1)

	# ## set voltages directly ###
	# print('start')
	# for i in range(16):
	# 	q.v[i] = 3.14
	# 	raw_input('hit enter when done')
	# 	q.v[i] = 0.0
	# print('end')

	# ## can also use 'set_value' and 'get_value' command ###
	# q.set_value(0,'v',3.1)
	# print(q.get_value(0,'v'))
	# print(q.get_all_values('v'))
	# q.set_value(0,'v',0.0)
	# print('done')

	# ### can also set limits such that if over it will go to 0. ###
	# q.set_value(0,'vmax', 3.0)
	# q.v[0] = 2.9
	# raw_input('now try going over vmax')
	# q.v[0] = 3.1

	##
