###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to realize adaptive feedforward (AFF) with the Iterative Learning 
Control (ILC) algorithm. The algorithm is tested with a cavity simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_sim import *
from rf_calib import * 
from rf_sysid import *
from rf_control import *

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
status1, Arfd, Brfd, Crfd, Drfd, _ = ss_discrete(Arf, Brf, Crf, Drf, Ts, 
                                              method = 'zoh', 
                                              plot   = False)
status2, Abmd, Bbmd, Cbmd, Dbmd, _ = ss_discrete(Abm, Bbm, Cbm, Dbm, Ts, 
                                              method = 'bilinear', 
                                              plot   = False)

# ---------------------------------------------------------
# identify the cavity impulse response
# ---------------------------------------------------------
# theoritical impulse response from the cavity equation
status, h = cav_impulse(wh, dw, Ts, order = 2000)

# identify the impulse response using data with least-square method
# here we simulate the data with the cavity simulator, in practice, the data can
# be collected from a real cavity
nwf = 200                                       # number of waveforms
U   = np.zeros((nwf, simN), dtype = complex)    # cavity input waveforms
Y   = np.zeros((nwf, simN), dtype = complex)    # cavity output waveforms

for i in range(nwf):
    # random input
    U[i] = np.random.normal(0.0, 1.0, simN) + 1j * np.random.normal(0.0, 1.0, simN)

    # simulate the cavity output
    status, _, Y[i], _ = sim_ncav_pulse(Arf, Brf, Crf, Drf, U[i], Ts)
    
status, h2 = iden_impulse(U, Y, order = 2000)

# plot (h and h2 has a difference of a factor 2, because in cav_impulse function,
# we normalized the cavity gain to 1, but for the cavity here, the gain is 2)
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.abs(h))
plt.plot(np.abs(h2), '--')
plt.xlabel('No. of Sample')
plt.ylabel('Impul. Resp. Ampl.')
plt.grid()
plt.subplot(1,2,2)
plt.plot(np.angle(h, deg=True))
plt.plot(np.angle(h2, deg=True), '--')
plt.grid()
plt.xlabel('No. of Sample')
plt.ylabel('Impul. Resp. Phas. (deg)')
plt.show(block = False)

# get the ILC matrix (change the value of P and Q for weighting the error or drive)
M = t_fill + t_flat                 # dimension of the matrix
status, L = AFF_ilc_design(h*2, M, P = np.eye(M)*50.0, Q = np.eye(M)*1.0)

# ---------------------------------------------------------
# simulate ILC
# ---------------------------------------------------------
vc2  = np.zeros(simN, dtype = complex)          # cavity output, V
vaff = np.zeros(simN, dtype = complex)          # feedforward signal, V

for pul_id in range(10):                        # check results with different iterations
    print('Simulate pulse {}'.format(pul_id))

    # initialize the states for cavity simulator and controller
    state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex)
    state_bm = np.matrix(np.zeros(Bbmd.shape), dtype = complex)

    # simulate a pulse
    for i in range(simN):
        # update the cavity output
        status, vc2[i], _, state_rf, state_bm = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vaff[i], state_rf,
                                                              Abmd      = Abmd, 
                                                              Bbmd      = Bbmd, 
                                                              Cbmd      = Cbmd, 
                                                              Dbmd      = Dbmd, 
                                                              vb_step   = vb[i],
                                                              state_bm0 = state_bm)
    # ILC
    vff_cor  = AFF_ilc(vc_sp[:M] - vc2[:M], L)  # feedforward correction with ILC
    vaff[:M] = vaff[:M] + vff_cor               # update the feedforward

# plot the results
plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc_sp),      label = 'Setpoint')
plt.plot(T, np.abs(vc1),  '--', label = 'Feedforward response')
plt.plot(T, np.abs(vc2),  '-.', label = 'AFF response')
plt.plot(T, np.abs(vaff), ':',  label = 'AFF drive')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc_sp,deg = True),       label = 'Setpoint')
plt.plot(T, np.angle(vc1,  deg = True), '--', label = 'Feedforward response')
plt.plot(T, np.angle(vc2,  deg = True), '-.', label = 'AFF response')
plt.plot(T, np.angle(vaff, deg = True), ':',  label = 'AFF drive')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.show(block = False)





















