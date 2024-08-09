###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to identify the cavity parameters based on the cavity equation. In
this example, we assumed a superconducting cavity (i.e., with time-varying half-
bandwidth and detuning) with beam loading in present
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_calib import * 
from llrflibs.rf_sysid import *
from llrflibs.rf_noise import *
from llrflibs.rf_misc import *

# parameters
pi = np.pi                              # short version of pi
fs = 1e6                                # sampling frequency, Hz
Ts = 1.0 / fs                           # sampling time, s

# read the data
'''
keys: 'dw_pul', 'halfbw', 'vc', 'vd', 'vdbeam', 'wh_pul'
'''
data    = load_mat('data_cavid_beam.mat')
vc      = data['vc']                    # complex waveform of cavity probe, arb. units
vd      = data['vd']                    # complex waveform of RF drive (already calibrated), arb. units
vb      = data['vdbeam']                # complex waveform of beam drive (already calibrated), arb. units
half_bw = data['halfbw']                # half-bandwidth derived from the pulse decay, rad/s
wh_pul0 = data['wh_pul']                # half-bandwidth within the RF pulse, rad/s
dw_pul0 = data['dw_pul']                # detuning within the RF pulse, rad/s

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(vc), label = 'Probe')
plt.plot(np.abs(vd), label = 'RF Drive')
plt.plot(np.abs(vb), label = 'Beam Drive')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(np.angle(vc, deg = True), label = 'Probe')
plt.plot(np.angle(vd, deg = True), label = 'RF Drive')
plt.plot(np.angle(vb, deg = True), label = 'Beam Drive')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.show(block = False)

# calculate the intra-pulse half bandwidth and detuning with beam and compare with the given values
# (set vb = None to see the results if we have no idea what the beam drive is)
status, wh_pul, dw_pul, vc_est, _ = cav_par_pulse_obs(vc, vd, half_bw, Ts, 
                                                      vb = vb, 
                                                      pole_scale = 80)

plt.figure()
plt.subplot(2,2,1)
plt.plot(abs(vc),           label = 'Cavity probe')
plt.plot(abs(vc_est), '--', label = 'Cavity probe denoised by ADRC')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,2,2)
plt.plot(np.angle(vc, deg = True),           label = 'Cavity probe')
plt.plot(np.angle(vc_est, deg = True), '--', label = 'Cavity probe denoised by ADRC')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.subplot(2,2,3)
plt.plot(wh_pul0[20:],      label = 'Given value by data')
plt.plot(wh_pul[20:], '--', label = 'Calculated value')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Half-bandwidth (rad/s)')
plt.subplot(2,2,4)
plt.plot( dw_pul0[20:]/2/pi,      label = 'Given value by data')    # the data already gave the correct detuning
plt.plot(-dw_pul[20:]/2/pi, '--', label = 'Calculated value')       # the minus sign is explained in "example_calib_for_ref_sc_cavity"
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Detuning (Hz)')
plt.show(block = False)

# estimate the beam drive 
# Note: it is not possible to identify the beam drive and the intra-pulse
#       time-varying half-bandwidth & detuning simultaneously. We have to 
#       know one of them to calculate the other
status, vb_est, vc_est1, _ = cav_beam_pulse_obs(vc, vd, wh_pul, -dw_pul, half_bw, Ts, 
                                                pole_scale =50)

plt.figure()
plt.subplot(2,2,1)
plt.plot(abs(vc),           label = 'Cavity probe')
plt.plot(abs(vc_est), '--', label = 'Cavity probe denoised by ADRC')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,2,2)
plt.plot(np.angle(vc, deg = True),           label = 'Cavity probe')
plt.plot(np.angle(vc_est, deg = True), '--', label = 'Cavity probe denoised by ADRC')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.subplot(2,2,3)
plt.plot(abs(vb),           label = 'Given value by data')
plt.plot(abs(vb_est), '--', label = 'Calculated value')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Beam Drive Amplitude (arb. units)')
plt.subplot(2,2,4)
plt.plot(np.angle(vb, deg = True),           label = 'Given value by data')
plt.plot(np.angle(vb_est, deg = True), '--', label = 'Calculated value')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Beam Drive Phase (deg)')
plt.show(block = False)




