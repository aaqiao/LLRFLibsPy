###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calibrate the virtual probe as a vector sum of the cavity forward
and reflected signals. The cavity probe signal should be available for the 
calibration here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import * 
from rf_sysid import *
from rf_noise import *
from rf_misc import *

# read the data
'''
dict_keys(['ans', 'roQ', 'Z0', 'f0', 'C', 'start', 'filling', 'flattop', 'beam_on', 
           'beam_off', 'ii', 'Vc', 'Vfor', 'Vref', 'Vkly', 'Vsum', 'Vdac', 'vct', 
           'vfort', 'vreft', 'cno', 'vc', 'vfor', 'vref', 'a', 'b', 'c', 'd', 'vdt', 'vd'])
'''
data = load_mat('data_directivity.mat')

vc   = data['vc']           # complex waveform of cavity probe
vf_m = data['vfor']         # complex waveform of forward signal
vr_m = data['vref']         # complex waveform of reflected signal

# align the forward and reflected signals according to the probe signal
vf_m = np.roll(vf_m, 0)
vr_m = np.roll(vr_m, 0)

cal_ids = 250               # time window for calibration as sample index
cal_ide = 1500

status, m, n = calib_vprobe(vc[cal_ids:cal_ide],
                            vf_m[cal_ids:cal_ide],
                            vr_m[cal_ids:cal_ide])
vc_cal = m * vf_m + n * vr_m

# plot
plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(np.abs(vc),           label = 'Probe')
plt.plot(np.abs(vc_cal), '--', label = 'Virtual Probe')
plt.plot(np.abs(vf_m),         label = 'Forward Raw Meas.')
plt.plot(np.abs(vr_m),         label = 'Reflected Raw Meas.')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(1,2,2)
plt.plot(np.angle(vc, deg = True),           label = 'Probe')
plt.plot(np.angle(vc_cal, deg = True), '--', label = 'Virtual Probe')
plt.plot(np.angle(vf_m, deg = True),         label = 'Forward Raw Meas.')
plt.plot(np.angle(vr_m, deg = True),         label = 'Reflected Raw Meas.')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.show(block = False)






