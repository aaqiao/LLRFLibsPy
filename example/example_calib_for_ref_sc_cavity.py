###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to calibrate the forward and reflected signals of a superconducting 
cavity (i.e., with time-varying half-bandwidth and detuning within the pulse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *
from llrflibs.rf_control import *
from llrflibs.rf_calib import * 
from llrflibs.rf_sysid import *
from llrflibs.rf_noise import *

# parameters
pi = np.pi                          # short version of pi
fs = 1e6                            # sampling frequency, Hz
Ts = 1.0 / fs                       # sampling time, s

# ---------------------------------------------------------
# calibrate the forward and reflected signals
# ---------------------------------------------------------
# read the data
'''
dict_keys(['roQ', 'Z0', 'f0', 'C', 'start', 'filling', 'flattop', 'beam_on', 
           'beam_off', 'ii', 'Vc', 'Vfor', 'Vref', 'Vkly', 'Vsum', 'Vdac', 
           'vct', 'vfort', 'vreft', 'cno', 'vc', 'vfor', 'vref', 'a', 'b', 'c', 
           'd', 'vdt', 'vd'])
'''
data = load_mat('data_directivity.mat')
vc   = data['vc']                   # complex probe waveform
vf_m = data['vfor']                 # complex forward waveform (raw meas.)
vr_m = data['vref']                 # complex reflected waveform (raw meas.)

# move the forward and reflected signal to align with cavity probe
vf_m = np.roll(vf_m, 0)
vr_m = np.roll(vr_m, 0)

# estimate the cavity drive (theoritical)
status, vc_f, b, a = notch_filt(vc, fnotch = 200e3, Q = 4, fs = fs)                                  # notch filter the passband mode
status, wh = half_bw_decay (np.abs(vc_f),               decay_ids = 1300, decay_ide = 1350, Ts = Ts) # half-bandwidth at decay
status, dw = detuning_decay(np.angle(vc_f, deg = True), decay_ids = 1300, decay_ide = 1350, Ts = Ts) # detuning at decay
status, vf_est, vr_est = cav_drv_est(vc_f, wh, Ts, detuning = dw)                                    # theoritical forward/reflected

# calibrate the forward and reflected
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

# plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(vc_f),          label = 'Probe')
plt.plot(np.abs(vf_m), ':',     label = 'Forward raw meas.')
plt.plot(np.abs(vf_cal),        label = 'Forward calibrated')
plt.plot(np.abs(vf_est), '--',  label = 'Forward estimated')
plt.plot(np.abs(vr_m), ':',     label = 'Reflected raw meas.')
plt.plot(np.abs(vr_cal),        label = 'Reflected calibrated')
plt.plot(np.abs(vr_est), '--',  label = 'Reflected estimated')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Amplitude (arb. units)')
plt.subplot(2,1,2)
plt.plot(np.angle(vc_f, deg = True),           label = 'Probe')
plt.plot(np.angle(vf_m, deg = True), ':',      label = 'Forward raw meas.')
plt.plot(np.unwrap(np.angle(vf_cal)) * 180/pi, label = 'Forward calibrated')
plt.plot(np.angle(vf_est, deg = True), '--',   label = 'Forward estimated')
plt.plot(np.angle(vr_m, deg = True), ':',      label = 'Reflected raw meas.')
plt.plot(np.unwrap(np.angle(vr_cal)) * 180/pi, label = 'Reflected calibrated')
plt.plot(np.angle(vr_est, deg = True), '--',   label = 'Reflected estimated')
plt.legend()
plt.grid()
plt.xlabel('Sample Id')
plt.ylabel('Phase (deg)')
plt.show(block = False)

# ---------------------------------------------------------
# calculate the half-bandwidth and detuning (compare two methods) within the pulse
# ---------------------------------------------------------
status, wh_pul0, dw_pul0 = cav_par_pulse(vc, vf_cal, wh, Ts)
status, wh_pul, dw_pul, vc_est, _ = cav_par_pulse_obs(vc, vf_cal, wh, Ts, 
                                                      pole_scale = 50)

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
plt.plot(wh_pul0,      label = 'Direct digital derivative')
plt.plot(wh_pul, '--', label = 'ADRC observer')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Half-bandwidth (rad/s)')
plt.subplot(2,2,4)
plt.plot(-dw_pul0/2/pi,      label = 'Direct digital derivative')
plt.plot(-dw_pul/2/pi, '--', label = 'ADRC observer')
plt.grid()
plt.legend()
plt.xlabel('Sample Id')
plt.ylabel('Detuning (Hz)')
plt.show(block = False)

'''
Note: in the detuning plot, a minus is added. This is because when measuring the
      data, the LO frequency is higher than the RF, causing the IF phase rotation
      direction opposite to the RF phase - this was an imperfection in the data
      used for the calculation
'''




