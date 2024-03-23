###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calculate the noise power spectral density (PSD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import * 
from rf_sysid import *
from rf_noise import *
from rf_det_act import *

# parameters
pi  = np.pi                 # shorter pi
fs  = 249.9e6               # sampling frequency, Hz
bit = 16                    # number of bits of the data
n   = 6                     # non-I/Q parameter: n samples cover m IF cycles
m   = 1

# get the data
data_t = load_mat('data_ref_sample.mat')
data   = data_t['data']
signal = data[:, 2]         # take No. 0,2,4,6 for valid signals

# calculate the PSD with different methods
result1 = calc_psd_coherent(signal, 
                            fs      = fs, 
                            bit     = bit, 
                            n_noniq = n, 
                            plot    = True)    # for coherent sampling
result2 = calc_psd(signal, 
                   fs   = fs, 
                   bit  = bit, 
                   plot = True)                # for general sampling (with blackman windowing)

plt.figure()
plt.plot(result1['freq'], result1['amp_resp'], label = 'Coherent')
plt.plot(result2['freq'], result2['amp_resp'], label = 'Windowing')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dBFS/Hz)')
plt.grid()
plt.legend()
plt.suptitle('PSD of ADC Raw Waveform')
plt.show(block = False)  

# calculate the amplitude and phase noise
status1, I, Q     = noniq_demod(signal, n, m)       # non-I/Q demodulation
status2, A, P_deg = iq2ap_wf(I, Q)                  # derive the amplitude and phase

A     = A[n:]                                       # remove the incomplete demod points
P_deg = P_deg[n:]

amp_noise = (A - np.mean(A)) / np.mean(A)           # relative amplitude noise
pha_noise = (P_deg - np.mean(P_deg)) * pi / 180.0   # phase noise, rad

result_an = calc_psd_coherent(amp_noise, 
                              fs      = fs, 
                              bit     = 0,          # need to set to 0 for amplitude/phase noise
                              n_noniq = n, 
                              plot    = False)
result_pn = calc_psd_coherent(pha_noise, 
                              fs      = fs, 
                              bit     = 0, 
                              n_noniq = n, 
                              plot    = False)

plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(amp_noise, label = 'Amplitude Noise (rel)')
plt.plot(pha_noise, label = 'Phase Noise (rad)')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Magnitude')
plt.subplot(1,2,2)
plt.plot(result_an['freq'], result_an['amp_resp'], label = 'Amplitude Noise (dB/Hz)')
plt.plot(result_pn['freq'], result_pn['amp_resp'], label = 'Phase Noise (dBrad^2/Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()
plt.suptitle('Amplitude & Phase Noise Time Series and PSD')
plt.show(block = False)


