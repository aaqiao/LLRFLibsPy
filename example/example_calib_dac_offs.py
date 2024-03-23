###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calibrate the DAC offset to remove the carrier leakage of I/Q
modulator when it is used for direct upconversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import * 
from rf_misc import *

# read the data
'''
dict_keys(['dac_I', 'dac_Q', 'iqm_I_old', 'iqm_Q_old', 'iqm_I_new', 'iqm_Q_new', 
           'leak_ids', 'leak_ide', 'sig_ids', 'sig_ide', 'delay'])
'''
data = load_mat('data_dac_offs.mat')

dac_I       = data['dac_I']                 # DAC output waveform, real part
dac_Q       = data['dac_Q']                 # DAC output waveform, imaginary part
iqm_I_old   = data['iqm_I_old']             # I/Q modulator output waveform, real part (before calibration)
iqm_Q_old   = data['iqm_Q_old']             # I/Q modulator output waveform, imaginary part (before calibration)
iqm_I_new   = data['iqm_I_new']             # I/Q modulator output waveform, real part (after calibration)
iqm_Q_new   = data['iqm_Q_new']             # I/Q modulator output waveform, imaginary part (after calibration)

# calculate the DAC offset correction
vdac = dac_I     + 1j*dac_Q                 # convert DAC output waveform to complex 
viqm = iqm_I_old + 1j*iqm_Q_old             # convert I/Q modulator output waveform to complex
 
leak_ids = data['leak_ids']                 # starting index for calculating leakage
leak_ide = data['leak_ide']                 # ending index for calculating leakage
sig_ids  = data['sig_ids']                  # starting index for calculating signal
sig_ide  = data['sig_ide']                  # ending index for calculating signal
delay    = data['delay']                    # delay in numer of points of I/Q modulator output compared to DAC 

status, offs_i, offs_q = calib_dac_offs(vdac, viqm, sig_ids, sig_ide, leak_ids, leak_ide, delay)    # call the routine

print('Offset should be added to DACs are {} and {}'.format(offs_i, offs_q))

# plot
plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(dac_I, label = 'DAC I')
plt.plot(dac_Q, label = 'DAC Q')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Magnitude (arb. units)')
plt.subplot(1,2,2)
plt.plot(iqm_I_old,       label = 'I/Q mod. output I (before calib)')
plt.plot(iqm_Q_old,       label = 'I/Q mod. output Q (before calib)')
plt.plot(iqm_I_new, '--', label = 'I/Q mod. output I (after calib.)')
plt.plot(iqm_Q_new, '--', label = 'I/Q mod. output Q (after calib.)')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Magnitude (arb. units)')
plt.show(block = False)




