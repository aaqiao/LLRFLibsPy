###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to demo the system identification with the ETFE method with PRBS 
input signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_sim import *
from rf_calib import * 
from rf_sysid import *
from rf_control import *

# ---------------------------------------------------------
# define the cavity simulation parameters
# ---------------------------------------------------------
f_scale = 5                         # sampling frequency scaling factor from the default 1 MHz
pi      = np.pi                     # shorter pi

# cavity parameters
fs      = 1e6 * f_scale             # sampling frequency, Hz
Ts      = 1/fs                      # sampling time, s

f0      = 1.3e9                     # RF operating frequency, Hz
roQ     = 1036                      # r/Q of the cavity, Ohm
QL      = 3e6                       # loaded quality factor of the cavity
RL      = 0.5 * roQ * QL            # cavity loaded resistence (Linac convention), Ohm
wh      = pi*f0 / QL                # half-bandwidth of the cavity, rad/s
dw      = wh                        # detuning of the cavity, rad/s

pb_modes = {'freq_offs': [-800e3],              # offset frequencies of cavity passband modes, Hz
            'gain_rel':  [-1],                  # gain of passband modes compared to the pi-mode
            'half_bw':   [2*np.pi*216 * 0.5]}   # half-bandwidth of the passband modes, rad/s

# ---------------------------------------------------------
# define the cavity model
# ---------------------------------------------------------
# cavity model (change plot to True for showing the frequency response)
result = cav_ss(wh, detuning = dw, passband_modes = pb_modes, plot = True)
Arf    = result[1]                  # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]

# ---------------------------------------------------------
# identify the cavity frquency response and compare with the model above
# ---------------------------------------------------------
# parameters for system identification
r   = 20                # number of periods of the inputs
N   = 2048              # batch size of the inputs for each period

# define the PRBS input with r periods
status, u_batch = prbs(N-1, -1.0, 1.0)
u = np.tile(u_batch, r)
    
# simulate the cavity output
status, _, y, _ = sim_ncav_pulse(Arf, Brf, Crf, Drf, u, Ts)

# add some noise to y
y += np.random.normal(0.0, 0.001, y.shape)

# estimate the frequency response
status, f, G, ufft, yfft = etfe(u, y, 
                                r = r, 
                                exclude_transient = True, 
                                transient_batch_num = 1, 
                                fs = fs)
# plot the results
plt.figure()
plt.subplot(2,2,1)
plt.plot(u, label = 'u')
plt.plot(y, label = 'y')
plt.xlabel('Sample Id')
plt.ylabel('Magnitude (arb. units)')
plt.grid()
plt.legend()
plt.subplot(2,2,3)
plt.plot(f, 20.0 * np.log10(np.abs(ufft)), label = 'Spec Input')
plt.plot(f, 20.0 * np.log10(np.abs(yfft)), label = 'Spec Output')
plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Estimated TF Mag. (dB)')
plt.subplot(2,2,2)
plt.plot(f, 20.0 * np.log10(np.abs(G)))
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Estimated TF Mag. (dB)')
plt.subplot(2,2,4)
plt.plot(f, np.angle(G, deg = True))
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Estimated TF Phase (deg)')
plt.show(block = False)










