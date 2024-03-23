###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calibrate the system gain and system phase of a standing-wave
cavity control loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import * 
from rf_sysid import *
from rf_noise import *
from rf_misc import *

# parameters
pi = np.pi          # shorter pi
fs = 1e6            # sampling frequency, Hz
Ts = 1.0 / fs       # sampling time, s

# read the data
'''
dict_keys(['roQ', 'Z0', 'f0', 'C', 'start', 'filling', 'flattop', 'beam_on', 
           'beam_off', 'ii', 'Vc', 'Vfor', 'Vref', 'Vkly', 'Vsum', 'Vdac', 'vct', 
           'vfort', 'vreft', 'cno', 'vc', 'vfor', 'vref', 'a', 'b', 'c', 'd', 'vdt', 'vd'])
'''
data = load_mat('data_directivity.mat')                 # data saved from FLASH, DESY

# DESY's LO frequency is wrong, causing a wrong phase rotation direction, we conjugate the signals
vc   = np.conj(data['vc'])                              # complex waveform of cavity probe signal
vf_m = np.conj(data['vfor'])                            # complex waveform of cavity forward signal (raw meas.)
vr_m = np.conj(data['vref'])                            # complex waveform of cavity reflected signal (raw meas.)
vact = data['Vdac']                                     # complex waveform of LLRF controller actuation (DAC output)
vact = vact * np.max(np.abs(vc)) / np.max(np.abs(vact)) # scale its magnitude similar to cavity probe

# plot the raw data
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(vc),   label = 'Probe')
plt.plot(np.abs(vf_m), label = 'Forward Raw Meas.')
plt.plot(np.abs(vr_m), label = 'Reflected Raw Meas.')
plt.plot(np.abs(vact), label = 'Actuation (scaled)')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(np.angle(vc,   deg = True), label = 'Probe')
plt.plot(np.angle(vf_m, deg = True), label = 'Forward Raw Meas.')
plt.plot(np.angle(vr_m, deg = True), label = 'Reflected Raw Meas.')
plt.plot(np.angle(vact, deg = True), label = 'Actuation (scaled)')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.suptitle('Cavity Raw Signals')
plt.show(block = False)

# align the timing of the forward and reflected signal according to the probe
vf_m = np.roll(vf_m, 0)
vr_m = np.roll(vr_m, 0)

# estimate the cavity drive (theoritical)
status, vc_f, b, a = notch_filt(vc, fnotch = 200e3, Q = 4, fs = fs)                                  # remove the passband mode
status, wh = half_bw_decay (np.abs(vc_f),               decay_ids = 1300, decay_ide = 1350, Ts = Ts) # half-bandwidth at decay, rad/s
status, dw = detuning_decay(np.angle(vc_f, deg = True), decay_ids = 1300, decay_ide = 1350, Ts = Ts) # detuning at decay, rad/s
status, vf_est, vr_est = cav_drv_est(vc_f, wh, Ts, detuning = dw)                                    # estimate the cavity drive and reflected

# calibrate the forward and reflected signals
status, a, b, c, d = calib_scav_for_ref(vc, vf_m, 
                                        vr_m      = vr_m, 
                                        pul_ids   = 1200, 
                                        pul_ide   = 1300, 
                                        decay_ids = 1310,
                                        decay_ide = 1550,
                                        half_bw   = wh, 
                                        Ts        = Ts,
                                        detuning  = dw)

vf_cal = a * vf_m + b * vr_m
vr_cal = c * vf_m + d * vr_m

# plot the calibrated signals
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(vc_f),          label = 'Probe')
plt.plot(np.abs(vf_cal),        label = 'Forward calibrated')
plt.plot(np.abs(vf_est), '--',  label = 'Forward estimated')
plt.plot(np.abs(vr_cal),        label = 'Reflected calibrated')
plt.plot(np.abs(vr_est), '--',  label = 'Reflected estimated')
plt.plot(np.abs(vact),          label = 'Actuation (scaled)')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(np.angle(vc_f, deg = True),            label = 'Probe')
plt.plot(np.unwrap(np.angle(vf_cal)) * 180/pi,  label = 'Forward calibrated')
plt.plot(np.angle(vf_est, deg = True), '--',    label = 'Forward estimated')
plt.plot(np.unwrap(np.angle(vr_cal)) * 180/pi,  label = 'Reflected calibrated')
plt.plot(np.angle(vr_est, deg = True), '--',    label = 'Reflected estimated')
plt.plot(np.angle(vact, deg = True),            label = 'Actuation (scaled)')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.suptitle('Calibrated Cavity Signals')
plt.show(block = False)

# calculate the system phase and system gain
status, sys_gain, sys_phase = calib_sys_gain_pha(vact, vf_cal)

plt.figure()
plt.subplot(1,2,1)
plt.plot(sys_gain)
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('System Gain')
plt.subplot(1,2,2)
plt.plot(sys_phase)
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('System Phase (deg)')
plt.show(block = False)






