import serial

class Arroyo5240(object):
    def __init__(self, port='COM4'):
        self.ser = serial.Serial(port)
        self.ser.baudrate = 38400
        self.ser.bytesize = serial.EIGHTBITS
        self.ser.parity = serial.PARITY_NONE
        self.ser.stopbits = serial.STOPBITS_ONE

    def send_rcv(self, command):
        self.ser.write((command + '\n').encode())
        answer = self.ser.readline()
        return answer.strip().decode()

    def send(self, command):
        self.ser.write((command + '\n').encode())

    def reset(self):
        self.ser.write(b'*RST\n')

    def decrease_temp(self, steps):
        self.ser.write(('TEC:DEC ' + str(steps) + '\n').encode())

    def increase_temp(self, steps):
        command = 'TEC:INC ' + str(steps) + '\n'
        self.ser.write(command.encode())

    def get_temp(self):
        self.ser.write(b'TEC:T?\n')
        temp = self.ser.readline()
        return float(temp.strip())

    def set_temp(self, temp):
        self.ser.write(('TEC:T ' + str(temp) + '\n').encode())

    def get_set_temp(self):
        self.ser.write(b'TEC:T?\n')
        set_temp = self.ser.readline()
        return float(set_temp.strip())

    def get_step(self):
        self.ser.write(b'TEC:TSTEP?\n')
        step = self.ser.readline()
        return float(step.strip())

    def set_step(self, step):
        self.ser.write(('TEC:TSTEP ' + str(step) + '\n').encode())

    def set_temp_limits(self, t_hi, t_lo):
        self.ser.write(('TEC:LIM:THI ' + str(t_hi) + '\n').encode())
        self.ser.write(('TEC:LIM:TLO ' + str(t_lo) + '\n').encode())

    def autotune(self, test_temp):
        self.ser.write(('TEC:AUTOTUNE ' + str(test_temp) + '\n').encode())


if __name__ == '__main__':
    import time
    input = False
    # input = True
    obj = Arroyo5240()
    temp = 20
    print(obj.get_temp())
    if input:
        obj.set_temp(temp)
        while obj.get_temp() != temp:
            while obj.get_temp() !=temp :
                time.sleep(1)
                print(obj.get_temp())
        print(obj.get_temp())
