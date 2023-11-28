###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to generate noise time series from PSD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import * 
from rf_sysid import *
from rf_noise import *
from rf_det_act import *

# input vectors of offset frequencies and noise PSDs
freq_vector = np.array([ 10,     100,     1e3,    10e3,    100e3,   1e6,     5e6,     10e6])    # offset freq. from carrier, Hz
psd_vector  = np.array([-82.03, -107.07, -128.22, -135.86, -145.81, -148.15, -148.62, -164])    # noise PSDs (dBrad^2/Hz for phase noise)
#freq_vector = np.array([10, 100])
#psd_vector  = np.array([-118.5, -118.5])   # jitter can be calculated as 10**((-118.5 + 10*np.log10(fs/2))/20)

N  = 2**16-1                               # number of samples in the time series
fs = 238e6                                 # sampling frequency, Hz

status, noise_series, freq_p, pn_p = gen_noise_from_psd(freq_vector, psd_vector, fs, N) # generate the time series
result = calc_psd(noise_series, fs = fs)    # calculate PSD to verify

plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(noise_series)
plt.xlabel('Sample Id')
plt.ylabel('Noise Magnitude (arb. units)')
plt.grid()
plt.subplot(1,2,2)
plt.plot(result['freq'], result['amp_resp'], label = 'PSD of Generated Time Series')
plt.plot(freq_p, pn_p,                       label = 'Input PSD Points')
plt.xscale('log')
plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'DSB PSD $(dBrad^2/Hz)$')
plt.show(block = False)

print('RMS from time series = {:.3e}'.format(np.std(noise_series)))

# calculate the RMS value from spectrum
status, freq_p2, pn_p2, jitter_p2 = calc_rms_from_psd(freq_vector, psd_vector, fs/N, fs/2, fs, N)

plt.figure()
plt.subplot(2,1,1)
plt.plot(freq_p2, pn_p2)
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'DSB PSD $(dBrad^2/Hz)$')
plt.xscale('log')
plt.grid()
plt.subplot(2,1,2)
plt.plot(freq_p2, jitter_p2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Jitter (rad)')
plt.xscale('log')
plt.grid()
plt.show(block = False)





