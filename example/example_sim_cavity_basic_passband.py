###################################################################################
#  Copyright (c) 2024 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example compares the cavity models:
    - full model including both fundamental and passband modes
    - model of only passband modes
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

pb_modes = {'freq_offs': [-800e3, -3e6],                # offset frequencies of cavity passband modes, Hz
            'gain_rel':  [-1, 1],                       # gain of passband modes compared to the pi-mode
            'half_bw':   [np.pi*216, np.pi*210]}        # half-bandwidth of the passband modes, rad/s

half_bw  = pi*f0 / QL                           # half-bandwidth of the cavity, rad/s
detuning = half_bw                              # detuning of the cavity, rad/s
beta     = 1e4                                  # input coupling factor of the cavity

# generate the cavity model (set plot to True to show the frequency response)
result = cav_ss(half_bw, detuning = detuning, beta = beta, passband_modes = pb_modes, plot = True)
status = result[0]
Arf    = result[1]                              # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]
Abm    = result[5]                              # A,B,C,D of cavity model for beam drive
Bbm    = result[6]
Cbm    = result[7]
Dbm    = result[8]

# generate the model with only passband modes
status2, Apb, Bpb, Cpb, Dpb = cav_ss_passband(pb_modes)
ss_freqresp(Apb, Bpb, Cpb, Dpb, plot = True, plot_pno = 10000, plot_maxf = 4e6)






