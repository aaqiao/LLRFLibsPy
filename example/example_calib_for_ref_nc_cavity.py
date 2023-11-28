###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calibrate the forward and reflected signals of a normal-conducting 
cavity (i.e., with constant half-bandwidth and detuning within the pulse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_sim import *
from rf_control import *
from rf_calib import * 
from rf_sysid import *
from rf_noise import *

# parameters
pi       = np.pi                    # short version of pi
fs       = 249.9e6                  # sampling frequency, Hz
Ts       = 1.0 / fs                 # sampling time, s
cav_beta = 2.02                     # input coupling factor of the cavity

# get the Gun cavity signals
data   = load_mat('data_Gun_WFs.mat')
vc_amp = data['vc_amp']
vc_pha = data['vc_pha']
vf_amp = data['vf_amp']
vf_pha = data['vf_pha']
vr_amp = data['vr_amp']
vr_pha = data['vr_pha']

vc   = vc_amp * np.exp(1j * vc_pha * pi / 180.0)  # complex probe waveform
vf_m = vf_amp * np.exp(1j * vf_pha * pi / 180.0)  # complex forward waveform (raw meas.)
vr_m = vr_amp * np.exp(1j * vr_pha * pi / 180.0)  # complex reflected waveform (raw meas.)

# align the timing of the forward and reflected signals with the cavity probe
# signal. This is done by comparing the raw measurement vf_m, vr_m with the 
# theoritical estimation vf_est, vr_est given below by "cav_drv_est"
vf_m = np.roll(vf_m, 5)
vr_m = np.roll(vr_m, 1)

# estimate the cavity drive (theoritical)
status, vc_f, b, a = notch_filt(vc, fnotch = 15.9e6, Q = 4, fs = fs)            # notch filter the passband mode
status, wh = half_bw_decay (vc_amp, decay_ids = 940, decay_ide = 1050, Ts = Ts) # calculate half-bandwidth at pulse decay
status, dw = detuning_decay(vc_pha, decay_ids = 940, decay_ide = 1050, Ts = Ts) # calculate the detuning at pulse decay
status, vf_est, vr_est = cav_drv_est(vc_f, wh, Ts, dw, beta = cav_beta)         # estimate the theoritical forward/reflected signals

# calibrate the forward and reflected
status, a, b, c, d = calib_ncav_for_ref(vc, vf_m, vr_m, 
                                        pul_ids  = 600, 
                                        pul_ide  = 1100, 
                                        half_bw  = wh, 
                                        Ts       = Ts,
                                        detuning = dw, 
                                        beta     = cav_beta)
vf_cal = a * vf_m + b * vr_m
vr_cal = c * vf_m + d * vr_m

# if the reflected signal is not available, we can directly estimate the cavity
# forward signal with the function "calib_cav_for". Note that in this case,
# the cavity forward signal should be measured before the circulator (i.e., directly
# after the klystron), otherwise, the crosstalk from the reflected power will cause 
# errors (as shown here - the forward signal was measured after the circulator)
status, a2 = calib_cav_for(vc, vf_m, 
                           pul_ids  = 840, 
                           pul_ide  = 940, 
                           half_bw  = wh, 
                           Ts       = Ts, 
                           detuning = dw, 
                           beta     = cav_beta)
vf_cal2 = a2 * vf_m

# plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(vc_amp,                label = 'Probe')
plt.plot(vf_amp, ':',           label = 'Forward raw meas.')
plt.plot(np.abs(vf_cal),        label = 'Forward calibrated')
plt.plot(np.abs(vf_cal2),       label = 'Forward calibrated (no refl.)')
plt.plot(np.abs(vf_est), '--',  label = 'Forward estimated')
plt.plot(vr_amp, ':',           label = 'Reflected raw meas.')
plt.plot(np.abs(vr_cal),        label = 'Reflected calibrated')
plt.plot(np.abs(vr_est), '--',  label = 'Reflected estimated')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(vc_pha,                                label = 'Probe')
plt.plot(vf_pha, ':',                           label = 'Forward raw meas.')
plt.plot(np.unwrap(np.angle(vf_cal)) * 180/pi,  label = 'Forward calibrated')
plt.plot(np.unwrap(np.angle(vf_cal2)) * 180/pi, label = 'Forward calibrated (no refl.)')
plt.plot(np.angle(vf_est, deg = True), '--',    label = 'Forward estimated')
plt.plot(vr_pha, ':',                           label = 'Reflected raw meas.')
plt.plot(np.unwrap(np.angle(vr_cal)) * 180/pi,  label = 'Reflected calibrated')
plt.plot(np.angle(vr_est, deg = True), '--',    label = 'Reflected estimated')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.suptitle('Calibration of Forward/Reflected Signals')
plt.show(block = False)

# ---------------------------------------------------------
# based on the calculation above, we construct the cavity state-space equation
# and simulate its output and compare with the actual measurement of the probe
# signal and the calibrated reflected signal
# ---------------------------------------------------------
# generate the RF Gun cavity model (change plot to True to see the freq. response)
result = cav_ss(wh, detuning = dw, beta = cav_beta, passband_modes = None, plot = True)
status = result[0]
Arf    = result[1]          # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]

# test the pulse response of the cavity
status, T, vc2, vr2 = sim_ncav_pulse(Arf, Brf, Crf, Drf, vf_cal, Ts)

plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc),         label = 'Probe')
plt.plot(T, np.abs(vc2), '--',  label = 'Probe sim')
plt.plot(T, np.abs(vr_cal),     label = 'Reflected')
plt.plot(T, np.abs(vr2), '--',  label = 'Reflected sim')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc,  deg = True),       label = 'Probe')
plt.plot(T, np.angle(vc2, deg = True), '--', label = 'Probe sim')
plt.plot(T, np.angle(vr_cal,  deg = True),   label = 'Reflected')
plt.plot(T, np.angle(vr2, deg = True), '--', label = 'Reflected sim')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.suptitle('Validation with Cavity Equation')
plt.show(block = False)


