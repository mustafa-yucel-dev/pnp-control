import qontrol
import sys
import time
import numpy as np

# Setup Qontroller
# serial_port_name = "/dev/tty.usbserial-FT31EUVZ"
serial_port_name = "COM3"
q = qontrol.QXOutput(serial_port_name = serial_port_name, response_timeout = 0.1)
q.vmax = 10
print ("Qontroller '{:}' initialised with firmware {:} and {:} channels".format(q.device_id, q.firmware, q.n_chs) )
i_max = 14
ivals = [float(x) for x in np.linspace(0, i_max, 30)]
ch = 46

print("Would you like to proceed? (y/n)")

value_in = "Y"
oi = [46]
if (value_in.upper() == "Y"):
    for ch in oi:
        q.i[ch] = 10

        time.sleep(0.2)
        v = q.v[ch]
        i = q.i[ch]
        print(f"channel {ch} voltage {v} current {i}, resistance {v/i*1000}")
    q.v[:] = 0
    q.close()
else:
	# Close the device
	q.close()
	sys.exit("Example complete.")
