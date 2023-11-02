import pyvisa as visa
import time

WL_MIN = 1480  # nm
WL_MAX = 1640  # nm
WL_STEP_MIN = 1e-4  # nm
WL_STEP_MAX = 160  # nm
SWEEP_SPEED_MIN = 0.5  # nm/s
SWEEP_SPEED_MAX = 100  # nm/s
SWEEP_SPEED_STEP = 0.1  # nm/s


class SantecTSL570(object):

    def __init__(self, address='GPIB0::19::INSTR'):
        rm = visa.ResourceManager()
        self.instr = rm.open_resource(address)
        self.name = self.instr.query('*IDN?')
        self.wl0 = self.get_wavelength()

    def test(self):
        # Initiates an instrument self-test and returns the results.
        status = self.instr.query('*TST?')

        if int(status) == 0:
            print('No error.')
        elif int(status) == 1:
            print('LD temperature is out of range (at LD OFF)')
        elif int(status) == 2:
            print('LD temperature is out of range (at LD ON)')
        elif int(status) == 4:
            print('Wavelength monitor temperature is out of range')
        elif int(status) == 8:
            print('LD injection current is overload')
        elif int(status) == 16:
            print('Power monitor is malfunction')

    def reset(self):
        self.instr.write('*RST')

    def setup_basic(self):
        # set basic settings
        self.instr.write('SOUR:COHCtrl')  # enable coherency ctrl
        self.set_output(output=True)
        self.set_power_unit('dBm')
        self.set_wav_unit('nm')
        self.set_wavelength(1550)

        # confirm basic settings
        if int(self.instr.query('SOUR:COHCtrl?')) == 1:
            print('Coherency control enabled.')
        else:
            print('Coherency control disabled.')
        print('Power Unit', self.get_power_unit())
        print('Wavelength Unit', self.get_wav_unit())
        print('Center Wavelength', self.get_wavelength())

    def enable_output_trigger(self,trigType=3,trigStep=1):
        """ Trig types: 1) Start 2) Stop 3) Step """
        self.instr.write('TRIG:OUTP '+str(trigType))
        if .0001 < trigStep < 160:
            self.instr.write('TRIG:OUTP:STEP '+str(trigStep))
        else:
            print('ERROR: STEP OUT OF BOUNDS')
            return 0

    def set_power_unit(self, power_unit):
        """ Should be either dBm, mW, or uW """
        if power_unit == 'dBm':
            val = '0'
        elif power_unit == 'mW':
            val = '1'
        else:
            return 'Invalid power unit.'

        self.instr.write('SOUR:POW:UNIT ' + val)

    def coh_ctrl_off(self):
        self.instr.write('SOUR:COHC 0')

    def coh_ctrl_on(self):
        self.instr.write('SOUR:COHC 1')

    def get_power_unit(self):
        unit_int = int(self.instr.query('SOUR:POW:UNIT?'))
        if unit_int == 0:
            return 'dBm'
        elif unit_int == 1:
            return 'mW'
        else:
            print('ERROR READING POWER UNIT.')
            return 0

    def get_wav_unit(self):
        unit_int = int(self.instr.query('SOUR:WAV:UNIT?'))
        if unit_int == 0:
            return 'nm'
        elif unit_int == 1:
            return 'THz'
        else:
            print('ERROR READING WAVELENGTH UNIT.')
            return 0

    def set_wav_unit(self, unit):
        # default to nm
        wav_unit_code = '0'
        # change to THz, if necessary
        if unit == 'THz':
            wav_unit_code = '1'
        # write unit
        self.instr.write('SOUR:WAV:UNIT ' + wav_unit_code)

    def get_wavelength(self):
        wavelength_str = self.instr.query('SOUR:WAV?')
        return float(wavelength_str)

    def set_wavelength(self, wav):
        # round to fourth decimal place
        wav = str(round(wav, 4))

        # delay = abs(float(wav) - float(self.wl0)) / SWEEP_SPEED_STEP
        delay = .1

        # set lambda
        self.instr.write(f'SOUR:WAV {wav}nm')
        time.sleep(delay)

        self.wl0 = wav

    def set_power(self, val, power_unit='dBm'):
        """
        Set power.
        """

        # hundredths resolution
        val = round(val, 2)

        # set and verify units and ranges
        if self.get_power_unit() is power_unit:
            if ((val >= -20.) and (val <= 13.)) is False:
                print('INVALID POWER LEVEL (-20->10 dBm).')
                return 0
        else:
            # correct the units
            self.set_power_unit(power_unit)
            print('Unit set to', self.get_power_unit())

            # verify range
            if ((val >= .01) and (val <= 10.)) is False:
                print('INVALID POWER LEVEL (.01->10 mW).')
                return 0

        # set power
        self.instr.write(f'SOUR:POW {val}')

    def get_power(self):
        """
        Read power.
        """

        return self.instr.query('SOUR:POW?')

    def get_wav_fine(self):
        """
        Read wavelength fine tuning setting.
        Range is -100->100
        """
        return self.instr.query('SOUR:WAV:FIN?')

    def set_wav_fine(self, val):
        """
        Read wavelength fine tuning setting.
        Range is -100->100
        """
        print('WARNING: THIS DOES NOT FUNCTIN')

        # hundredths precision
        val = round(val, 2)

        # if in valid input range
        if (val >= -100) and (val <= 100):
            self.instr.write('SOUR:WAV:FIN %f' % (val))
        else:
            print('INVALID FINE TUNING SETTING.')
            return 0

    def disable_wav_fine(self):
        """
        Disable fine tuning of the wavelength setting.
        """
        self.instr.write('SOUR:WAV:FIN:DIS')

    def set_output(self, output=False):
        """
        Enable or disable laser diode.
        """

        if output is True:
            self.instr.write('SOUR:POW:STAT 1')
        else:
            self.instr.write('SOUR:POW:STAT 0')

    def get_output(self):
        """
        Read laser diode on/off status.
        """
        if int(self.instr.query('SOUR:POW:STAT?')) == 0:
            return False
        else:
            return True

    def runDefault():
        time.sleep(.01)

    def open_shutter(self):
        """
        Open internal shutter of device.
        """
        self.instr.write('SOUR:POW:SHUT 0')

    def close_shutter(self):
        """
        Close internal shutter of device.
        """
        self.instr.write('SOUR:POW:SHUT 1')

    def wavelength_sweep(self, wl_start, wl_stop, pwr, mode, speed, delay=0, step=0, runFxn = runDefault):
        """
        Perform a wavelength sweep of the laser.
        Mode register:
        0: step, one-way
        1: cont, one-way
        2: step, two-way
        3: cont, two-way
        """

        # start wavelength
        wl_start = round(wl_start, 4)
        if (wl_start < WL_MIN) or (wl_start > WL_MAX):
            print('ERROR: WAVELENGTH OOB.')
            return 0
        else:
            self.instr.write(f'SOUR:WAV:SWE:STAR {wl_start}nm')

        # stop wavelength
        wl_stop = round(wl_stop, 4)
        if (wl_stop < WL_MIN) or (wl_stop > WL_MAX):
            print('ERROR: WAVELENGTH OOB.')
            return 0
        else:
            self.instr.write(f'SOUR:WAV:SWE:STOP {wl_stop}nm')

        # power
        self.set_power(pwr)

        # mode
        if mode not in [0, 1, 2, 3]:
            print('ERROR: INVALID MODE.')
            return 0
        else:
            self.instr.write(f'SOUR:WAV:SWE:MOD {mode}')
        if mode in [0, 2]: #get step if needed
            if (step < WL_STEP_MIN) or (step > WL_STEP_MAX):
                print('ERROR: INVALID STEP.')
                return 0
            else:
                step = round(step, 4)
                self.instr.write(f'SOUR:WAV:SWE:STEP {step}')

        # speed
        if (speed < SWEEP_SPEED_MIN) or (speed > SWEEP_SPEED_MAX):
            print('ERROR: SWEEP SPEED.')
            return 0
        else:
            self.instr.write(f'SOUR:WAV:SWE:SPE {speed}')

        # start sweep
        print('Starting sweep...')
        self.instr.write('SOUR:WAV:SWE:STAT 1')

        # pause to give time for sweep to start
        time.sleep(.1)

        # wait until the sweep is done
        while(int(self.instr.query('SOUR:WAV:SWE?')) != 0):
            runFxn()

        print('Sweep complete!')

        return 1

        # extract data
#        data = self.instr.query('SOUR:READ:DAT')

#        return data


class SantecLaser(SantecTSL570):
    pass

if __name__ == '__main__':
    obj = SantecTSL570('GPIB0::19::INSTR')
    obj.setup_basic()
    print(obj.get_wavelength())
