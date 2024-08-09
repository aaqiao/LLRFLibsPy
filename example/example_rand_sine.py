###################################################################################
#  Copyright (c) 2024 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to generate random sine series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_noise import *

N       = 32768         # number of points
fs      = 1e6           # sampling frequency, Hz
nfreq   = 3             # number of frequencies
Amin    = 1             # min/max amplitude
Amax    = 10
fmin    = 1e3           # min/max frequency, Hz
fmax    = 200e3 

# generate the series
status, data, t = rand_sine(N, fs, 
                    nfreq = nfreq, 
                    Amin  = Amin, 
                    Amax  = Amax, 
                    fmin  = fmin, 
                    fmax  = fmax)

# plot the series
plt.figure()
plt.plot(t, data)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.grid()
plt.show(block = False)

# calculate the PSD
result = calc_psd(data, 
                  fs   = fs, 
                  bit  = 1, 
                  plot = True)



