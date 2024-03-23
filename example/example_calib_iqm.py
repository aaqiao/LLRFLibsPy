###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calibrate the I/Q modulator imbalance (displaying the data tested at 
SwissFEL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import * 
from rf_misc import *

# read the data
'''
dict_keys(['dac_A', 'dac_P', 'iqm_A_old', 'iqm_P_old', 'iqm_A_new', 'iqm_P_new', 'invM'])
'''
data = load_mat('data_iqm_imbal.mat')

dac_A       = data['dac_A']                 # DAC scan points, amplitude
dac_P       = data['dac_P']                 # DAC scan points, phase in deg
iqm_A_old   = data['iqm_A_old']             # I/Q modulator output points, amplitude (before calib)
iqm_P_old   = data['iqm_P_old']             # I/Q modulator output points, phase in deg (before calib)
iqm_A_new   = data['iqm_A_new']             # I/Q modulator output points, amplitude (after calib)
iqm_P_new   = data['iqm_P_new']             # I/Q modulator output points, phase in deg (after calib)

# demonstrate the calibration
_, invM, _, _ = calib_iqmod(dac_A     * np.exp(1j * dac_P     * np.pi / 180.0), 
                            iqm_A_old * np.exp(1j * iqm_P_old * np.pi / 180.0))
print(invM)

# get the I and Q of the signals for plotting in an I-Q plot
dac_I       = dac_A     * np.cos(dac_P     * np.pi / 180.0)
dac_Q       = dac_A     * np.sin(dac_P     * np.pi / 180.0)
iqm_I_old   = iqm_A_old * np.cos(iqm_P_old * np.pi / 180.0)
iqm_Q_old   = iqm_A_old * np.sin(iqm_P_old * np.pi / 180.0)
iqm_I_new   = iqm_A_new * np.cos(iqm_P_new * np.pi / 180.0)
iqm_Q_new   = iqm_A_new * np.sin(iqm_P_new * np.pi / 180.0)

# unwrap the phase error caused by imbalance of the I/Q modulator
P1t  = np.unwrap(iqm_P_old, period = 360.0) - dac_P
P1t -= np.mean(P1t)
P2t  = np.unwrap(iqm_P_new, period = 360.0) - dac_P
P2t -= np.mean(P2t)

# plot the amplitude and phase error before/after calibration when scanning the DAC drive
# in a full circle
plt.figure()
plt.subplot(2,2,1)
plt.plot(dac_I, dac_Q, '*')
plt.xlabel('DAC I')
plt.ylabel('DAC Q')
plt.grid()
plt.subplot(2,2,2)
plt.plot(iqm_I_old, iqm_Q_old, '*', label = 'Before calib')
plt.plot(iqm_I_new, iqm_Q_new, 'o', label = 'After calib')
plt.xlabel('I/Q Mod. Out I')
plt.ylabel('I/Q Mod. Out Q')
plt.legend()
plt.grid()
plt.subplot(2,2,3)
plt.plot(dac_P, iqm_A_old, label = 'Before calib')
plt.plot(dac_P, iqm_A_new, label = 'After calib')
plt.xlabel('DAC Out Phase (deg)')
plt.ylabel('I/Q Mod. Out Amplitude')
plt.legend()
plt.grid()
plt.subplot(2,2,4)
plt.plot(dac_P, P1t, label = 'Before calib')
plt.plot(dac_P, P2t, label = 'After calib')
plt.xlabel('DAC Out Phase (deg)')
plt.ylabel('I/Q Mod. Out Phase Error (deg)')
plt.legend()
plt.grid()
plt.show(block = False)










