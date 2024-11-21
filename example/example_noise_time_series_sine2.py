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

ts    = np.linspace(0, (N-1)/fs, N)                             # time array of pulse 1, s
ts2   = ts + 30e-3                                              # time array of pulse 2, s
data  = np.zeros(N)
data2 = np.zeros(N)
for i in range(n_sine):
    data  = data  + amplts[i] * np.sin(2 * np.pi * freqs[i] * ts  + phases[i])
    data2 = data2 + amplts[i] * np.sin(2 * np.pi * freqs[i] * ts2 + phases[i])

result1 = calc_psd(data,  fs = fs)                              # calculate PSDs to verify
result2 = calc_psd(data2, fs = fs)

plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(data,  label = 'Pulse 1')
plt.plot(data2, label = 'Pulse 2')
plt.xlabel('Sample Id')
plt.ylabel('Noise Magnitude (arb. units)')
plt.grid()
plt.subplot(1,2,2)
plt.plot(result1['freq'], result1['amp_resp'], label = 'PSD of Generated Time Series (Pulse 1)')
plt.plot(result2['freq'], result2['amp_resp'], label = 'PSD of Generated Time Series (Pulse 2)')
plt.plot(freq_vector, psd_vector, '*', label = 'Input PSD Points')
plt.plot(freqs, psds, label = 'Intepreted PSD Points')
plt.plot()
plt.xscale('log')
plt.grid()
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'DSB PSD $(dBrad^2/Hz)$')
plt.show(block = False)

'''
Note: if we use multiple sine waves to emulate the amplitude or phase noise, we can 
      make continous-time simulation of how the long-term noise changes. For example,
      for pulsed RF stations, we can use this method to implement a noise generator,
      which can generate noise time series not only for inside a pulse, but also 
      continously for many pulses (simply put the continous time array of different
      pulses into the sine wave functions).
'''











