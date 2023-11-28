###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to realize adaptive feedforward (AFF) with time-reversed low-pass 
filter. The algorithm is tested with a cavity simulator
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

# ---------------------------------------------------------
# define the cavity control loop simulation parameters
# ---------------------------------------------------------
f_scale = 1                         # sampling frequency scaling factor from the default 1 MHz
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

# ---------------------------------------------------------
# define the cavity model
# ---------------------------------------------------------
# cavity model (change plot to True for showing the frequency response)
result = cav_ss(wh, detuning = dw, passband_modes = None, plot = False)

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
status1, Arfd, Brfd, Crfd, Drfd = ss_discrete(Arf, Brf, Crf, Drf, Ts, 
                                              method = 'zoh', 
                                              plot   = False)
status2, Abmd, Bbmd, Cbmd, Dbmd = ss_discrete(Abm, Bbm, Cbm, Dbm, Ts, 
                                              method = 'bilinear', 
                                              plot   = False)

# ---------------------------------------------------------
# define the general feedback controller
# ---------------------------------------------------------
Kp = 10                                 # proportional gain
Ki = 1e5                                # integral gain

# basic RF controller (set plot to True to show the frequency response)
status, Akc, Bkc, Ckc, Dkc = basic_rf_controller(Kp, Ki,
                                                 notch_conf = None, 
                                                 plot       = True, 
                                                 plot_maxf  = 4e3)

# get the discrete controller
status, Akd, Bkd, Ckd, Dkd = ss_discrete(Akc, Bkc, Ckc, Dkc, Ts, 
                                         method     = 'bilinear', 
                                         plot       = False,
                                         plot_pno   = 10000)

# ---------------------------------------------------------
# simulate the feedback loop for each pulse / AFF for multiple pulses
# ---------------------------------------------------------
vc2     = np.zeros(simN, dtype = complex)       # closed-loop cavity output, V
vf2     = np.zeros(simN, dtype = complex)       # overall cavity input, V
vfb     = np.zeros(simN, dtype = complex)       # cavity input induced by feedback, V
vaff    = np.zeros(simN, dtype = complex)       # cavity input induced by adaptive feedforward, V
vf_all  = 0.0 + 1j*0.0                          # overall cavity input of a time step, V

plt.figure()
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc_sp),  label = 'Setpoint')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc_sp,deg = True), label = 'Setpoint')
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')

for pul_id in range(10):
    print('Simulate pulse {}'.format(pul_id))

    # initialize the states for cavity simulator and controller
    state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex) # state of cavity model for RF drive
    state_bm = np.matrix(np.zeros(Bbmd.shape), dtype = complex) # state of cavity model for beam drive
    state_k  = np.matrix(np.zeros(Bkd.shape),  dtype = complex) # state of the feedback controller

    # simulate a pulse
    for i in range(simN):
        # update the cavity output for a time step
        status, vc2[i], _, state_rf, state_bm = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_all, state_rf,
                                                              Abmd      = Abmd, 
                                                              Bbmd      = Bbmd, 
                                                              Cbmd      = Cbmd, 
                                                              Dbmd      = Dbmd, 
                                                              vb_step   = vb[i],
                                                              state_bm0 = state_bm)
       
        # simple controller for a time step
        status, vf_all, vfb[i], state_k = control_step(Akd, Bkd, Ckd, Dkd, vc_sp[i]-vc2[i], state_k, ff_step = vaff[i])    

        # clear the drive when out of the pulse
        if i >= t_fill + t_flat:
            vf_all = vfb[i] = 0.0

        # remember the overall input
        vf2[i] = vf_all

    # update the AFF
    status, vff_cor = AFF_timerev_lpf(vfb, fs/150, fs)
    vaff += 0.5 * vff_cor

    # remember the closed-loop output without AFF (first pulse)
    if pul_id == 0:
        vc_fb_only = vc2.copy()

    # plot for this pulse
    plt.subplot(2,1,1)
    plt.plot(T, np.abs(vc2), label = 'Iteration ' + str(pul_id+1))
    plt.subplot(2,1,2)
    plt.plot(T, np.angle(vc2, deg = True),  label = 'Iteration ' + str(pul_id+1))

plt.subplot(2,1,1)
plt.legend()
plt.subplot(2,1,2)
plt.legend()
plt.show(block = False)

# plot the results
plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc_sp),            label = 'Setpoint')
plt.plot(T, np.abs(vc1),        '--', label = 'Feedforward response')
plt.plot(T, np.abs(vc_fb_only), '-.', label = 'Feedback response')
plt.plot(T, np.abs(vc2),        ':',  label = 'Feedback + AFF response')
plt.plot(T, np.abs(vaff),       '--', label = 'AFF drive')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc_sp,deg = True),             label = 'Setpoint')
plt.plot(T, np.angle(vc1,  deg = True),       '--', label = 'Feedforward response')
plt.plot(T, np.angle(vc_fb_only, deg = True), '-.', label = 'Feedback response')
plt.plot(T, np.angle(vc2,  deg = True),       ':',  label = 'Feedback + AFF response')
plt.plot(T, np.angle(vaff, deg = True),       '--', label = 'AFF drive')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.show(block = False)





















