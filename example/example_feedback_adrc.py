###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to test the ADRC RF feedback controller with a cavity simulator.
You can compare the results with "example_feedback_basic" under the same 
conditions (beam loading, disturbance, etc.)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *
from llrflibs.rf_control import *
from llrflibs.rf_calib import * 
from llrflibs.rf_sysid import *
from llrflibs.rf_noise import *

# ---------------------------------------------------------
# define the cavity control loop simulation parameters
# ---------------------------------------------------------
f_scale = 10                        # sampling frequency scaling factor from the default 1 MHz
                                    # here we need larger sampling frequency for accurate discretization
                                    # of the ADRC observer which has faster dynamics compared to the
                                    # closed-loop cavity - a too slow sampling rate cannot discretize such
                                    # fast dynamics accurately
pi      = np.pi                     # shorter pi
simN    = 2048 * f_scale            # number of points in the waveform for simulating an RF pulse

# cavity parameters
fs      = 1e6 * f_scale             # sampling frequency, Hz
Ts      = 1/fs                      # sampling time, s

f0      = 1.3e9                     # RF operating frequency, Hz
roQ     = 1036                      # r/Q of the cavity, Ohm
QL      = 3e6                       # loaded quality factor of the cavity
RL      = 0.5 * roQ * QL            # cavity loaded resistence (Linac convention), Ohm
ig      = 0.016                     # RF drive power equivalent current, A
ib      = 0.008                     # average beam current, A
t_fill  = 510 * f_scale             # length of cavity filling period, sample
t_flat  = 800 * f_scale             # length of cavity flattop period, sample
t_bms   = 600 * f_scale             # time when the beam pulse starts, sample
t_bme   = 1000 * f_scale            # time when the beam pulse stops, sample

vc0     = 25e6                      # flattop cavity voltage, V
wh      = pi*f0 / QL                # half-bandwidth of the cavity, rad/s
dw      = wh                        # detuning of the cavity, rad/s

pb_modes = {'freq_offs': [-800e3],              # offset frequencies of cavity passband modes, Hz
            'gain_rel':  [-1],                  # gain of passband modes compared to the pi-mode
            'half_bw':   [2*np.pi*216 * 0.5]}   # half-bandwidth of the passband modes, rad/s

# derive the set point and the feedforward
status, vc_sp, vf_ff, vb, T = cav_sp_ff(wh, t_fill, t_flat, Ts, 
                                        vc0      = vc0, 
                                        detuning = 0, 
                                        pno      = simN)

vb = np.zeros(simN, dtype = complex)            # initialize the beam drive voltage, V
vb[t_bms:t_bme] = -RL * ib

# add a harmonic disturbance in the feedforward
dist_freq = 5e3                                         # disturbance frequency offset, Hz
vf_dist   = vc0 * 0.5 * np.cos(2 * pi * dist_freq * T)  # disturbance waveform, V
vf_ff     = vf_ff + vf_dist                             # apply the disturbance to cavity input via FF

# ---------------------------------------------------------
# define the cavity model
# ---------------------------------------------------------
# cavity model (change plot to True for showing the frequency response)
result = cav_ss(wh, detuning = dw, passband_modes = pb_modes, plot = False)

status = result[0]
Arf    = result[1]                  # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]
Abm    = result[5]                  # A,B,C,D of cavity model for beam drive
Bbm    = result[6]
Cbm    = result[7]
Dbm    = result[8]

# response of the cavity to the feedforward given above
status, T1, vc1, vr1 = sim_ncav_pulse(Arf, Brf, Crf, Drf, vf_ff, Ts,
                                        Abmc = Abm,
                                        Bbmc = Bbm,
                                        Cbmc = Cbm,
                                        Dbmc = Dbm,
                                        vb   = vb)

# discretize the cavity equation (change plot to True to plot freq. response)
status1, Arfd, Brfd, Crfd, Drfd, _ = ss_discrete(Arf, Brf, Crf, Drf, Ts, 
                                              method = 'zoh', 
                                              plot   = False)
status2, Abmd, Bbmd, Cbmd, Dbmd, _ = ss_discrete(Abm, Bbm, Cbm, Dbm, Ts, 
                                              method = 'bilinear', 
                                              plot   = False)

# ---------------------------------------------------------
# define the basic feedback controller
# ---------------------------------------------------------
Kp = 30                                 # proportional gain
Ki = 1e5                                # integral gain

notches = {'freq_offs': [5e3],          # notch filter config for FB
           'gain':      [1000],
           'half_bw':   [2*np.pi*20]}

# basic RF controller (set plot to True to show the frequency response)
status, Akc, Bkc, Ckc, Dkc = basic_rf_controller(Kp, Ki,
                                                 notch_conf = notches,      # set to None to see results without notch filters
                                                 plot       = True, 
                                                 plot_maxf  = 10e3)

# get the discrete controller (set plot to True to show the freq. responses)
status, Akd, Bkd, Ckd, Dkd, _ = ss_discrete(Akc, Bkc, Ckc, Dkc, Ts, 
                                         method     = 'bilinear', 
                                         plot       = True,
                                         plot_pno   = 10000)

# ---------------------------------------------------------
# define the ADRC controller and discretize it
# ---------------------------------------------------------
status, Aobc, Bobc, Cobc, Dobc, b0 = ADRC_controller(wh, pole_scale = 200)
status, Aobd, Bobd, _, _, _ = ss_discrete(Aobc, Bobc, Cobc, Dobc, Ts, method = 'bilinear')

# ---------------------------------------------------------
# simulate the feedback loop
# ---------------------------------------------------------
vc2     = np.zeros(simN, dtype = complex)                   # closed-loop cavity voltage waveform, V
vf2     = np.zeros(simN, dtype = complex)                   # overall cavity drive, V
vc2_est = np.zeros(simN, dtype = complex)                   # estimated cavity voltage by ADRC, V
f       = np.zeros(simN, dtype = complex)                   # estimated general disturbance by ADRC

state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex) # state of cavity model of RF response
state_bm = np.matrix(np.zeros(Bbmd.shape), dtype = complex) # state of cavity model of beam response
state_k  = np.matrix(np.zeros(Bkd.shape),  dtype = complex) # state of the basic controller
state_ob = np.matrix(np.zeros((2,1)),      dtype = complex) # state of the ADRC observer
vf_all = 0.0 + 1j*0.0                                       # overall cavity drive of the time step

use_ADRC = True

for i in range(simN):
    # remember the overall input
    vf2[i] = vf_all

    # update the cavity output for the time step
    status, vc2[i], _, state_rf, state_bm = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_all, state_rf,
                                                          Abmd      = Abmd, 
                                                          Bbmd      = Bbmd, 
                                                          Cbmd      = Cbmd, 
                                                          Dbmd      = Dbmd, 
                                                          vb_step   = vb[i],
                                                          state_bm0 = state_bm)
    # apply the ADRC/basic controller
    if use_ADRC:
        status, vf_all, _, state_k, state_ob, vc2_est[i], f[i] = ADRC_control_step(Akd, Bkd, Ckd, Dkd,
                                                                    Aobd, Bobd, b0, vc_sp[i], vc2[i], vf2[i], 
                                                                    state_k, state_ob, 
                                                                    vf_step = vf_ff[i if i == 0 else i-1], ff_step = vf_ff[i], 
                                                                    apply_to_err = True)   # set to True if the controller is not simple P control
    else:        
        # as comparison, simple controller (not stable because the passband modes => ADRC can even filter out the passband modes)
        status, vf_all, _, state_k = control_step(Akd, Bkd, Ckd, Dkd, vc_sp[i]-vc2[i], state_k, ff_step = vf_ff[i])

    # clear the drive when out of the pulse
    vf_all = 0 if (i >= t_fill + t_flat) else vf_all

# plot the results
plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc_sp),        label = 'Setpoint')
plt.plot(T, np.abs(vc1), '--',    label = 'Feedforward response')
plt.plot(T, np.abs(vc2), '-.',    label = 'Closed-loop response')
plt.plot(T, np.abs(vc2_est), ':', label = 'ADRC est. cavity voltage')
plt.plot(T, np.abs(vf2), '--',    label = 'Overall cavity drive')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc_sp,deg = True),          label = 'Setpoint')
plt.plot(T, np.angle(vc1,  deg = True), '--',    label = 'Feedforward response')
plt.plot(T, np.angle(vc2,  deg = True), '-.',    label = 'Closed-loop response')
plt.plot(T, np.angle(vc2_est,  deg = True), ':', label = 'ADRC est. cavity voltage')
plt.plot(T, np.angle(vc2,  deg = True), '--',    label = 'Overall cavity drive')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.show(block = False)

'''
plt.figure()
plt.plot(T, np.real(f))
plt.plot(T, np.imag(f), '--')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Est. General Disturbance')
plt.show(block = False)
'''



















