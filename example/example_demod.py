###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to demonstrate RF signal demodulation with different methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_calib import * 
from llrflibs.rf_sysid import *
from llrflibs.rf_noise import *
from llrflibs.rf_det_act import *

# load the data
data = load_mat('data_adcraw_wfs.mat')

ref_raw = data['ref_raw']               # ADC raw waveform of RF reference 
vm_raw  = data['vm_raw']                # ADC raw waveform of vector mod. output
kly_raw = data['kly_raw']               # ADC raw waveform of klystron output
boc_raw = data['boc_raw']               # ADC raw waveform of BOC output
fs      = data['fs']                    # sampling frequency, Hz
fif     = data['f_if']                  # IF frequency, Hz
n       = data['noniq_n']               # non-I/Q parameters: n sample covers m IF cycles
m       = data['noniq_m']

# make the demodulation (you can change the raw waveforms to other signals to compare)
status, I, Q   = noniq_demod(boc_raw, n, m)                             # non-I/Q demodulation
status, I1, Q1 = twop_demod(boc_raw, fif, fs)                           # demodulation with every two samples
status, I2, Q2, Isig, Qsig, Iref, Qref = asyn_demod(boc_raw, ref_raw)   # asynchronous (LO/Clock) demodulation

plt.figure()
plt.subplot(2,1,1)
plt.plot(I,        label = 'Non I/Q result')
plt.plot(I1, '--', label = 'Two Sample result')
plt.plot(I2, '-.', label = 'Async result')
plt.plot(Isig,':', label = 'Async (signal)')
plt.plot(Iref,     label = 'Async (reference)')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('I Component (arb. units)')
plt.subplot(2,1,2)
plt.plot(Q,        label = 'Non I/Q result')
plt.plot(Q1, '--', label = 'Two Sample result')
plt.plot(Q2, '-.', label = 'Async result')
plt.plot(Qsig,':', label = 'Async (signal)')
plt.plot(Qref,     label = 'Async (reference)')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Q Component (arb. units)')
plt.suptitle('Comparison of Demod Methods')
plt.show(block = False)

# convert to A/P and compare with the self demod method based on Hilbert transform
status, A, P_deg = iq2ap_wf(I, Q)                   # conver the I/Q waveform to amplitude/phase
status, A3, P3   = self_demod_ap(boc_raw, n = 6)    # self-demodulation with Hilbert transform
status, Ar, Pr   = self_demod_ap(ref_raw, n = 6)    # self-demodulation with Hilbert transform
plt.figure()
plt.subplot(2,1,1)
plt.plot(A,        label = 'Non I/Q result')
plt.plot(A3, '--', label = 'Self demod of signal')
plt.plot(Ar, '-.', label = 'Self demod of reference')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(P_deg,                             label = 'Non I/Q result')
plt.plot(np.unwrap(P3, period = 360), '--', label = 'Self demod of signal')
plt.plot(np.unwrap(Pr, period = 360), '-.', label = 'Self demod of reference')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.suptitle('Self Demod with Hilbert Transform')
plt.show(block = False)









