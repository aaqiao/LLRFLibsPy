###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to generate noise time series (multile sine waves) from PSD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_noise import *

# phase noise specification
freq_vector = np.array([ 10,     100,     1e3,    10e3,    100e3,   1e6,     5e6,     10e6])    # offset freq. from carrier, Hz
psd_vector  = np.array([-82.03, -107.07, -128.22, -135.86, -145.81, -148.15, -148.62, -164])    # noise PSDs (dBrad^2/Hz for phase noise)
#freq_vector = np.array([10, 100])
#psd_vector  = np.array([-118.5, -118.5])    # jitter can be calculated as 10**((-118.5 + 10*np.log10(fs/2))/20)

N  = 2**16-1                                # number of samples in the time series
fs = 238e6                                  # sampling frequency, Hz

# generate with the IFFT method
status, noise_series, freq_p, pn_p = gen_noise_from_psd(freq_vector, psd_vector, fs, N) # generate the time series
result = calc_psd(noise_series, fs = fs)    # calculate PSD to verify

plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(noise_series)
plt.xlabel('Sample Id')
plt.ylabel('Noise Magnitude (arb. units)')
plt.grid()
plt.subplot(1,2,2)
plt.plot(result['freq'], result['amp_resp'], 
         label = 'PSD of Generated Time Series (IFFT)')
plt.plot(freq_p, pn_p, label = 'Input PSD Points')
plt.xscale('log')
plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'DSB PSD $(dBrad^2/Hz)$')
plt.show(block = False)

# generate with the random sine functions
n_sine = 2000                                                   # number of sine waves
f_st   = 1                                                      # start frequency, Hz                
f_ed   = fs/2                                                   # end frequency, Hz

freqs  = np.zeros(n_sine)                                       # select freqs, Hz
freqs[:1000] = np.arange(1000, dtype = float) + f_st
freqs[1000:] = np.linspace(freqs[999] + f_st, f_ed, 1000, 
                           endpoint = True) 
status, amplts, phases, psds = gen_rand_sine_from_psd(freq_vector, 
                                                     psd_vector, 
                                                     freqs)     # gen sine waves

ts   = np.linspace(0, (N-1)/fs, N)                              # time array, s
data = np.zeros(N)
for i in range(n_sine):
    data = data + amplts[i] * np.sin(2 * np.pi * freqs[i] * ts + phases[i])

result2 = calc_psd(data, fs = fs)                               # calculate PSD to verify

plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(data)
plt.xlabel('Sample Id')
plt.ylabel('Noise Magnitude (arb. units)')
plt.grid()
plt.subplot(1,2,2)
plt.plot(result2['freq'], result2['amp_resp'], 
         label = 'PSD of Generated Time Series (sine)')
plt.plot(freq_vector, psd_vector, '*', label = 'Input PSD Points')
plt.plot(freqs, psds, label = 'Intepreted PSD Points')
plt.plot()
plt.xscale('log')
plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'DSB PSD $(dBrad^2/Hz)$')
plt.show(block = False)

# compare the jitter
print(np.std(noise_series), np.std(data))

# calculate the RMS value from spectrum
status, freq_p2, pn_p2, jitter_p2 = calc_rms_from_psd(result['freq'][1:],   result['amp_resp'][1:],  fs/N, fs/2, fs, N)
status, freq_p3, pn_p3, jitter_p3 = calc_rms_from_psd(result2['freq'][1:],  result2['amp_resp'][1:], fs/N, fs/2, fs, N)

plt.figure()
plt.subplot(2,1,1)
plt.plot(freq_p2, pn_p2, label = 'PSD of Generated Time Series (IFFT)')
plt.plot(freq_p3, pn_p3, label = 'PSD of Generated Time Series (sine)')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'DSB PSD $(dBrad^2/Hz)$')
plt.xscale('log')
plt.grid()
plt.legend()
plt.subplot(2,1,2)
plt.plot(freq_p2, jitter_p2, label = 'Jitter Integral (IFFT)')
plt.plot(freq_p3, jitter_p3, label = 'Jitter Integral (sine)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Jitter (rad)')
plt.xscale('log')
plt.grid()
plt.legend()
plt.show(block = False)










