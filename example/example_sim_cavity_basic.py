###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to simulate the cavity response. In this simulation, we assumed
that the half-bandwidth and detuning of the cavity are constant. Since this 
is usually the case for normal-conducting cavities, the function names are 
denoted by "ncav" standing for normal-conducting cavities. However, the example 
used the parameters of TESLA cavity, but assuming it with constant half-bandwidth
and detuning. In reality, the detuning (sometime the half-bandwidth if there
is quench) of a superconducting cavity is often time-varying due to the Lorenz-
force detuning and microphonics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_sim import *
from rf_control import *
from rf_calib import *

# define the cavity parameters
pi      = np.pi                                 # shorter pi
fs      = 1e6                                   # sampling frequency, Hz
Ts      = 1 / fs                                # sampling time, s
N       = 2048                                  # number of points in the pulse

f0      = 1.3e9                                 # RF operating frequency, Hz
roQ     = 1036                                  # r/Q of the cavity, Ohm
QL      = 3e6                                   # loaded quality factor
RL      = 0.5 * roQ * QL                        # loaded resistance (Linac convention), Ohm
ig      = 0.016                                 # RF drive power equivalent current, A
ib      = 0.008                                 # average beam current, A
t_fill  = 510                                   # length of cavity filling period, sample
t_flat  = 1300                                  # end time of the flattop period, sample

pb_modes = {'freq_offs': [-800e3],              # offset frequencies of cavity passband modes, Hz
            'gain_rel':  [-1],                  # gain of passband modes compared to the pi-mode
            'half_bw':   [2*np.pi*216 * 0.5]}   # half-bandwidth of the passband modes, rad/s

half_bw  = pi*f0 / QL                           # half-bandwidth of the cavity, rad/s
detuning = half_bw                              # detuning of the cavity, rad/s
beta     = 1e4                                  # input coupling factor of the cavity

# generate the cavity model (set plot to True to show the frequency response)
result = cav_ss(half_bw, detuning = detuning, beta = beta, passband_modes = pb_modes, plot = False)
status = result[0]
Arf    = result[1]                              # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]
Abm    = result[5]                              # A,B,C,D of cavity model for beam drive
Bbm    = result[6]
Cbm    = result[7]
Dbm    = result[8]

# test the pulse response of the cavity for RF and beam drives
vf = np.zeros(N, dtype = complex)               # complex waveform of RF drive
vb = np.zeros(N, dtype = complex)               # complex waveform of beam drive

vf[:t_fill]       =  RL * ig                    # define the forward RF drive
vf[t_fill:t_flat] =  RL * ig
vb[t_fill:t_flat] = -RL * ib                    # define the beam drive

status, T, vc, vr = sim_ncav_pulse(Arf, Brf, Crf, Drf, vf, Ts,  # simulate the pulse response using cavity continous equation
                                   Abmc = Abm, 
                                   Bbmc = Bbm, 
                                   Cbmc = Cbm, 
                                   Dbmc = Dbm, 
                                   vb   = vb)

plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc),       label = 'Probe')
plt.plot(T, np.abs(vf), '--', label = 'RF Forward')
plt.plot(T, np.abs(vb), '-.', label = 'Beam drive')
plt.plot(T, np.abs(vr),       label = 'Reflected')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc, deg = True),       label = 'Probe')
plt.plot(T, np.angle(vf, deg = True), '--', label = 'RF Forward')
plt.plot(T, np.angle(vb, deg = True), '-.', label = 'Beam drive')
plt.plot(T, np.angle(vr, deg = True),       label = 'Reflected')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.suptitle('Cavity Response Simulated with Continuous State-space Model')
plt.show(block = False)

# discretize the state-space equation (set plot to True to show the frequency response)
status1, Arfd, Brfd, Crfd, Drfd, _ = ss_discrete(Arf, Brf, Crf, Drf, 
                                              Ts     = Ts, 
                                              method = 'zoh', 
                                              plot   = True)       # discretize the RF response model
status2, Abmd, Bbmd, Cbmd, Dbmd, _ = ss_discrete(Abm, Bbm, Cbm, Dbm, 
                                              Ts     = Ts, 
                                              method = 'bilinear', 
                                              plot   = True)       # discretize the beam response model

# simulate the cavity response per time step using the discrete models (we compare
# the results with the continous model outputs)
vc2 = np.zeros(N, dtype = complex)  # complex waveform of cavity probe simulated with discrete state-space model
vr2 = np.zeros(N, dtype = complex)  # complex waveform of cavity reflected ...
vc3 = np.zeros(N, dtype = complex)  # complex waveform of cavity probe simulated with Euler method discretized equation
vr3 = np.zeros(N, dtype = complex)  # complex waveform of cavity reflected ...

state_rf = np.matrix(np.zeros(Brfd.shape), dtype = complex) # state of the RF response state-space equation
state_bm = np.matrix(np.zeros(Bbmd.shape), dtype = complex) # state of the beam response state-space equation
state_vc = 0.0 + 1j*0                                       # state for Euler method-based discretization

for i in range(N):    
    # use discrete state-space equation
    result = sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf[i], state_rf,
                           Abmd      = Abmd, 
                           Bbmd      = Bbmd, 
                           Cbmd      = Cbmd, 
                           Dbmd      = Dbmd, 
                           vb_step   = vb[i],  
                           state_bm0 = state_bm)
    vc2[i]   = result[1]
    vr2[i]   = result[2]
    state_rf = result[3]
    state_bm = result[4]
   
    # use simple discrete equation based on the Euler method
    status, vc3[i], vr3[i] = sim_ncav_step_simple(half_bw, detuning, vf[i], vb[i], state_vc, Ts, 
                                                  beta = beta)    
    state_vc = vc3[i]

# plot
plt.figure();
plt.subplot(2,1,1)
plt.plot(T, np.abs(vc),        label = 'Cavity probe continous sim')
plt.plot(T, np.abs(vc2), '--', label = 'Cavity probe discrete sim SS')
plt.plot(T, np.abs(vc3), '-.', label = 'Cavity probe discrete sim Euler')
plt.plot(T, np.abs(vr),        label = 'Reflected continous sim')
plt.plot(T, np.abs(vr2), '--', label = 'Reflected discrete sim SS')
plt.plot(T, np.abs(vr3), '-.', label = 'Reflected discrete sim Euler')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.subplot(2,1,2)
plt.plot(T, np.angle(vc,  deg = True),       label = 'Cavity probe continous sim')
plt.plot(T, np.angle(vc2, deg = True), '--', label = 'Cavity probe discrete sim SS')
plt.plot(T, np.angle(vc3, deg = True), '-.', label = 'Cavity probe discrete sim Euler')
plt.plot(T, np.angle(vr,  deg = True),       label = 'Reflected continous sim')
plt.plot(T, np.angle(vr2, deg = True), '--', label = 'Reflected discrete sim SS')
plt.plot(T, np.angle(vr3, deg = True), '-.', label = 'Reflected discrete sim Euler')
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')
plt.suptitle('Cavity Response Simulated with Discrete Models')
plt.show(block = False)

# convert the forward voltage and reflected voltage to power
status, for_power, ref_power, C = for_ref_volt2power(roQ, QL, vf, vr)
plt.figure();
plt.plot(T, for_power/1000)
plt.plot(T, ref_power/1000)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Power (kW)')
plt.show(block = False)




