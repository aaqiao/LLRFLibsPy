###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to demo the system identification with the ETFE method with PRBS 
input signals. In this script, we identify the closed-loop response (sensitivity)
by comparing the same response of the cavity in closed and open loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *
from llrflibs.rf_calib import * 
from llrflibs.rf_sysid import *
from llrflibs.rf_control import *

# ---------------------------------------------------------
# define the cavity simulation parameters
# ---------------------------------------------------------
f_scale = 5                         # sampling frequency scaling factor from the default 1 MHz
pi      = np.pi                     # shorter pi

# cavity parameters
fs      = 1e6 * f_scale             # sampling frequency, Hz
Ts      = 1/fs                      # sampling time, s

f0      = 1.3e9                     # RF operating frequency, Hz
roQ     = 1036                      # r/Q of the cavity, Ohm
QL      = 3e6                       # loaded quality factor of the cavity
RL      = 0.5 * roQ * QL            # cavity loaded resistence (Linac convention), Ohm
wh      = pi*f0 / QL                # half-bandwidth of the cavity, rad/s
dw      = wh                        # detuning of the cavity, rad/s

pb_modes = {'freq_offs': [-800e3],              # offset frequencies of cavity passband modes, Hz
            'gain_rel':  [-1],                  # gain of passband modes compared to the pi-mode
            'half_bw':   [2*np.pi*216 * 0.5]}   # half-bandwidth of the passband modes, rad/s

vc_sp   = 25e6                      # setpoint
vf_ff   = vc_sp / 2                 # feedforward

# ---------------------------------------------------------
# define the cavity model
# ---------------------------------------------------------
# cavity model (change plot to True for showing the frequency response)
result = cav_ss(wh, detuning = dw, passband_modes = None, plot = False)
Arf    = result[1]                  # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]

# discretize the cavity equation (change plot to True to plot freq. response)
status, Arfd, Brfd, Crfd, Drfd, _ = ss_discrete(Arf, Brf, Crf, Drf, Ts, 
                                                method = 'zoh', 
                                                plot   = False)

# test the response of the cavity to feedforward
simN = 2048 * f_scale
vf   = np.zeros(simN) + vf_ff
s1, T1, vc1, vr1 = sim_ncav_pulse(Arf, Brf, Crf, Drf, vf, Ts)   # continuous model

vc2      = np.zeros(simN, dtype = complex)                      # discrete model
state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex)
for i in range(simN):
    s2, vc2[i], _, state_rf, _ = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_ff, state_rf)

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.abs(vf),    'b', label = 'Feedforward')
plt.plot(np.abs(vc1),   'r', label = 'Continuous model')
plt.plot(np.abs(vc2), '--g', label = 'Discrete model')
plt.grid(); plt.legend(); plt.xlabel('Time (sample)'); plt.ylabel('Amplitude (V)')
plt.subplot(2,1,2)
plt.plot(np.angle(vf,  deg = True),   'b', label = 'Feedforward')
plt.plot(np.angle(vc1, deg = True),   'r', label = 'Continuous model')
plt.plot(np.angle(vc2, deg = True), '--g', label = 'Discrete model')
plt.grid(); plt.legend(); plt.xlabel('Time (sample)'); plt.ylabel('Phase (deg)')
plt.show(block = False)

# ---------------------------------------------------------
# define the feedback controller
# ---------------------------------------------------------
Kp = 1000                               # proportional gain
Ki = 1e5                                # integral gain

# basic RF controller (set plot to True to show the frequency response)
status, Akc, Bkc, Ckc, Dkc = basic_rf_controller(Kp, Ki,
                                                 notch_conf = None,
                                                 plot       = False, 
                                                 plot_maxf  = 10e3)

# get the discrete controller (set plot to True to show the freq. responses)
status, Akd, Bkd, Ckd, Dkd, _ = ss_discrete(Akc, Bkc, Ckc, Dkc, Ts, 
                                         method     = 'bilinear', 
                                         plot       = False,
                                         plot_pno   = 10000)

# ---------------------------------------------------------
# identify the cavity frquency response and compare with the model above
# ---------------------------------------------------------
# parameters for system identification
r   = 20                # number of periods of the inputs
N   = 2048 * f_scale    # batch size of the inputs for each period
Ad  = 1e6               # PRBS magnitude

# define the PRBS input with r periods
status, u_batch = prbs(N-1, -1.0 * Ad, 1.0 * Ad)
u    = np.tile(u_batch, r)
simN = len(u)

# define the setpoint for closed-loop simulation
t_fill = 510 * f_scale  # length of cavity filling period, sample
t_flat = simN - t_fill  # length of cavity flattop period, sample
_, sp_wf, _, _, _ = cav_sp_ff(wh, t_fill, t_flat, Ts, 
                                        vc0      = vc_sp, 
                                        detuning = 0, 
                                        pno      = simN)

# simulate the cavity output for 4 scenarios with PRBS applyed to cavity input:
#   a. open-loop with only feedforward
#   b. open-loop with feedforward + PRBS
#   c. closed-loop with only feedforward
#   d. closed-loop with feedforward + PRBS   
vc_o1 = np.zeros(simN, dtype = complex)
vc_o2 = np.zeros(simN, dtype = complex)
vc_c1 = np.zeros(simN, dtype = complex)
vc_c2 = np.zeros(simN, dtype = complex)

state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex)
for i in range(simN):
    _, vc_o1[i], _, state_rf, _ = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_ff, state_rf)

state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex)
for i in range(simN):
    _, vc_o2[i], _, state_rf, _ = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_ff + u[i], state_rf)

state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex)
state_k  = np.matrix(np.zeros(Bkd.shape),  dtype = complex)
vf_all   = 0.0 + 1j*0.0
for i in range(simN):
    _, vc_c1[i], _, state_rf, _ = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_all, state_rf)
    _, vf_all, _, state_k = control_step(Akd, Bkd, Ckd, Dkd, sp_wf[i] - vc_c1[i], state_k, 
                                         ff_step = vf_ff)

state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex)
state_k  = np.matrix(np.zeros(Bkd.shape),  dtype = complex)
vf_all   = 0.0 + 1j*0.0
for i in range(simN):
    _, vc_c2[i], _, state_rf, _ = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_all, state_rf)
    _, vf_all, _, state_k = control_step(Akd, Bkd, Ckd, Dkd, sp_wf[i] - vc_c2[i], state_k, 
                                         ff_step = vf_ff + u[i])

plt.figure()
plt.plot(np.abs(sp_wf),   'b', label = 'Setpoint')
plt.plot(np.abs(vc_c1),   'r', label = 'Closed-loop with FF')
plt.plot(np.abs(vc_c2), '--g', label = 'Closed loop with FF + PRBS')
plt.grid(); plt.legend(); plt.xlabel('Time (sample)'); plt.ylabel('Amplitude (V)')
plt.show(block = False)

# get the incremental cavity outputs of open-/closed-loop
y_o = vc_o2 - vc_o1
y_c = vc_c2 - vc_c1

# estimate the frequency response for the same two points
s1, f1, G1, ufft1, yfft1 = etfe(u, y_o, 
                                r = r, 
                                exclude_transient = True, 
                                transient_batch_num = 1, 
                                fs = fs)
s2, f2, G2, ufft2, yfft2 = etfe(u, y_c, 
                                r = r, 
                                exclude_transient = True, 
                                transient_batch_num = 1, 
                                fs = fs)

# estimate the senstivity function
S = G2 / G1
S = S[f1 > 0]
f = f1[f1 > 0]
plt.figure()
plt.semilogx(f, 20.0 * np.log10(np.abs(S)))
plt.grid()
plt.xlabel('Freq (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Sensitivity Function')
plt.show(block = False)












