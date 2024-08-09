###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to derive the cavity setpoint and feedforward waveforms for given
cavity parameters, desired accelerating voltage/phase, and beam
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
ib0     = 0.003                     # average beam current, A
phib    = -30.0                     # beam phase (Linac convention), deg
t_fill  = 510 * f_scale             # length of cavity filling period, sample
t_flat  = 800 * f_scale             # length of cavity flattop period, sample
t_bms   = 600 * f_scale             # time when the beam pulse starts, sample
t_bme   = 1000 * f_scale            # time when the beam pulse stops, sample

vc0     = 25e6                      # flattop cavity voltage, V
wh      = pi*f0 / QL                # half-bandwidth of the cavity, rad/s
dw      = 2*pi*100                  # detuning of the cavity, rad/s

# derive the setpoint and the feedforward waveforms
status, vc_sp, vf_ff, vb, T = cav_sp_ff(wh, t_fill, t_flat, Ts, simN, detuning = dw, 
                                        vc0        = vc0,
                                        const_fpow = True,      # try both cases w/o constant filling power
                                        ib0        = ib0,
                                        phib_deg   = phib,
                                        beam_ids   = t_bms,
                                        beam_ide   = t_bme,
                                        roQ_or_RoQ = roQ,
                                        QL         = QL,
                                        machine    = 'linac')

# ---------------------------------------------------------
# simulate the cavity response
# ---------------------------------------------------------
# cavity model
result = cav_ss(wh, detuning = dw, plot = False)
Arf    = result[1]          # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]
Abm    = result[5]          # A,B,C,D of cavity model for beam drive
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

# plot the results
plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc_sp),       label = 'Setpoint')
plt.plot(T, np.abs(vc1),   '--', label = 'Feedforward response')
plt.plot(T, np.abs(vf_ff), '-.', label = 'Feedforward signal')
plt.plot(T, np.abs(vb),    ':',  label = 'Beam drive signal')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc_sp, deg = True),       label = 'Setpoint')
plt.plot(T, np.angle(vc1,   deg = True), '--', label = 'Feedforward response')
plt.plot(T, np.angle(vf_ff, deg = True), '-.', label = 'Feedforward signal')
plt.plot(T, np.angle(vb,    deg = True), ':',  label = 'Beam drive signal')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.show(block = False)











