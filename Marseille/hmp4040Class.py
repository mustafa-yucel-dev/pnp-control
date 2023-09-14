import pyvisa as visa
import numpy as np
import time

class HMP4040(object):
    def __init__(self, address):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address) 
        #self.set_voltage(1, 1)      
        self.instr.timeout = 5000
        #self.instr.read_termination = '\n'
        #self.instr.write_termination = '\n'
        self.name = self.instr.query('*IDN?')

        print('Connected to HMP4040.')
        if self.is_output_enabled():
            print('Power supply output is on.')
        else:
            print('Power supply output is off.')
        for ch in [1, 2, 3, 4]:
            vval = self.get_voltage(ch)
            ival = self.get_meas_current(ch)
            if self.get_output_state(ch):
                print('Channel %s: %s V set, %s A measured, output ENABLED' % (ch, float(vval), float(ival)))
            else:
                print('Channel %s: %s V set, %s A measured, output disabled' % (ch, float(vval), float(ival)))
        self.help()

    def help(self):
        print('''Here are the available functions to call: \n
                 set_voltage(channel, voltage), \n
                 get_voltage(channel), \n
                 set_current_limit(channel, current), \n
                 get_current_limit(channel), \n
                 set_output_state(channel, state), \n
                 get_output_state(channel), \n
                 enable_output(), \n
                 disable_output(), \n
                 is_output_enabled(), \n
                 get_meas_voltage(channel), \n
                 get_meas_current(channel)''')

    def reset(self):
        # reset the instrument
        self.instr.write('*RST')
        
    def select_channel(self, ch):
        retries = 5
        while retries > 0:   
            self.instr.write('INST OUT%s' % ch)
            inst_select = self.instr.query('INST?')
            if inst_select != 'OUTP%s\n' % ch:
                retries -= 1
            else:
                return
        print('The selected instrument, {}, is not the correct instrument!'.format(inst_select))
        raise

    def set_voltage(self, ch, voltage):
        self.select_channel(ch)
        retries = 5
        while retries > 0:
            self.instr.write('VOLT %s' % voltage)
            volts = self.instr.query('VOLT?')
            if float(volts) == round(voltage, 3):
                return
            else:
                retries -= 1
        print('The voltage at channel {} is not {} V, but {} V.'.format(ch, voltage, volts))
    
    def get_voltage(self, ch):
        self.select_channel(ch)
        return float(self.instr.query('VOLT?'))

    def set_current_limit(self, ch, current):
        self.select_channel(ch)
        retries = 5
        while retries > 0:
            self.instr.write('CURR %s' % current)
            curr = self.instr.query('CURR?')
            if float(curr) == round(current, 3):
                return
            else:
                retries -= 1
        print('The current at channel {} is not {} A, but {} A.'.format(ch, current, curr))
    
    def get_current_limit(self, ch):
        self.select_channel(ch)
        return float(self.instr.query('CURR?'))
    
    def set_output_state(self, ch, state):
        self.select_channel(ch)
        if state:
            self.instr.write('OUTP:SEL 1')
        else:
            self.instr.write('OUTP:SEL 0')
    
    def get_output_state(self, ch):
        self.select_channel(ch)
        return bool(float(self.instr.query('OUTP:SEL?')))
    
    def enable_output(self):
        self.instr.write('OUTP:GEN 1')

    def disable_output(self):
        self.instr.write('OUTP:GEN 0')
    
    def is_output_enabled(self):
        return bool(float(self.instr.query('OUTP:GEN?')))

    def get_meas_voltage(self, ch):
        self.select_channel(ch)
        return float(self.instr.query('MEAS:VOLT?'))

    def get_meas_current(self, ch):
        self.select_channel(ch)
        return float(self.instr.query('MEAS:CURR?'))
    
    def calc_num_of_volts(self, v_start, v_end, step):
        volt_diff = v_end-v_start
        if volt_diff >= 0:
            return (volt_diff//step)+1
        else:
            return abs((volt_diff//step)-1)
    
    def sweep_single_channel(self, v_start, v_end, step, pause, channel, zero_between_steps):
        '''Sweep the voltage values from v_start to v_end by taking steps of size 'step'.
        'Pause' is the pause time between consecutive steps, channel is the channel at which
        the sweep occurs, and zero_between_steps is a boolean indicating whether the voltage
        will be zero'ed first when taking a step. 
        '''
        num_of_volts = self.calc_num_of_volts(v_start, v_end, step)
        volts = np.linspace(v_start, v_end, num=num_of_volts)
        for volt in volts:
            if zero_between_steps:
                self.set_voltage(channel, 0)
                time.sleep(0.1) # time to ensure that the channel is actually set to 0
                self.set_voltage(channel, volt)
            else:
                self.set_voltage(channel, volt)
            time.sleep(pause)

    def sweep_multi_channel(self, v_start, v_end, step, pause, number_of_channels, zero_between_steps):
        '''Sweep the voltage across parallel-connected channels by dividing 'step' into 'number_of_channels'.
        '''
        num_of_volts = self.calc_num_of_volts(v_start, v_end, step)
        v_start_channel = v_start/number_of_channels
        v_end_channel = v_end/number_of_channels
        volts = np.linspace(v_start_channel, v_end_channel, num=num_of_volts)
        channels = list(range(1, number_of_channels+1))
        for volt in volts:
            for channel in channels:
                if zero_between_steps:
                    self.set_voltage(channel, 0)
                    time.sleep(0.1) # time to ensure that the channel is actually set to 0
                    self.set_voltage(channel, volt)
                else:
                    self.set_voltage(channel, volt)
            time.sleep(pause)


if __name__ == '__main__':

    hmp = HMP4040('ASRL4::INSTR')
    hmp.enable_output()
    hmp.sweep_multi_channel(21, 1, 5, 3, 4, True)
    hmp.disable_output()