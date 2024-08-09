###################################################################################
#  Copyright (c) 2024 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to simulate the cavity with mechanical modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *
from llrflibs.rf_control import *

# parameters
Ts  = 1e-6                  # electrical model sampling time, s
mds = 1                     # down-sampling for mech model
Tsm = Ts * mds              # mech model sampling time, s

# define the mechanical modes and descritize the model
mech_modes = {'f': [280, 341, 460, 487, 618],
              'Q': [40, 20, 50, 80, 100],
              'K': [2, 0.8, 2, 0.6, 0.2]}

status, Am, Bm, Cm, Dm = cav_ss_mech(mech_modes)
status, Ad, Bd, Cd, Dd, _ = ss_discrete(Am, Bm, Cm, Dm, 
                                     Ts     = Tsm, 
                                     method = 'zoh', 
                                     plot   = False,
                                     plot_pno = 10000)

# define a cavity
f0      = 1.3e9                                 # RF operating frequency, Hz
roQ     = 1036                                  # r/Q of the cavity, Ohm
QL      = 3e6                                   # loaded quality factor
wh      = np.pi * f0 / QL                       # half bandwidth, rad/s
RL      = 0.5 * roQ * QL                        # loaded resistance (Linac convention), Ohm
ig      = 0.016                                 # RF drive power equivalent current, A
ib      = 0.008                                 # average beam current, A
t_fill  = 510                                   # length of cavity filling period, sample
t_flat  = 1300                                  # end time of the flattop period, sample
dw0     = 2*np.pi*000                           # initial detuning, rad/s

N   = 2048 * 128
vc  = np.zeros(N, dtype = complex)
vf  = np.zeros(N, dtype = complex)
vr  = np.zeros(N, dtype = complex)
vb  = np.zeros(N, dtype = complex)
dw  = np.zeros(N) 

#vf[:t_fill]       =  RL * ig                    # define the forward RF drive
#vf[t_fill:t_flat] =  RL * ig
#vb[t_fill:t_flat] = -RL * ib                    # define the beam drive
vf[:] = RL * ig * 0.1

state_m  = np.matrix(np.zeros(Bd.shape))        # state of the mechanical equation
state_vc = 0.0                                  # state of cavity equation
dw_step0 = 0.0

for i in range(N):
    status, vc[i], vr[i], dw[i], state_m = sim_scav_step( wh, 
                                                          dw_step0,
                                                          dw0, 
                                                          vf[i], 
                                                          vb[i], 
                                                          state_vc, 
                                                          Ts, 
                                                          beta      = 1e4,
                                                          state_m0  = state_m, 
                                                          Am        = Ad, 
                                                          Bm        = Bd, 
                                                          Cm        = Cd, 
                                                          Dm        = Dd,
                                                          mech_exe  = True)           
    state_vc = vc[i]
    dw_step0 = dw[i]
    
# make the plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(abs(vc) * 1e-6)
plt.xlabel('Time (Ts)')
plt.ylabel('Cavity Voltage (MV)')
plt.subplot(2,1,2)
plt.plot(dw / 2 / np.pi)
plt.xlabel('Time (Ts)')
plt.ylabel('Detuning (Hz)')
plt.show()

