###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to estimate the required RF power (steady-state) for specific beam
loading and accelerating voltage/phase for standing-wave cavities 
This example is for a normal-conducting cavity in storage ring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_sim import *

# parameters
pi      = np.pi
f0      = 499.5935e6                # RF operating frequency, Hz
vc0     = 0.35e6                    # cavity voltage, V
ib0     = 0.4                       # beam average current, A
phib    = 147.556 - 90              # beam phase (convert to the Linac convention: 90 deg for on crest)
Q0      = 29909                     # unloaded quality factor
roQ     = 3.4e6 / Q0                # r/Q, Ohm
beta_vec= np.linspace(1, 8, 100)    # cavity input coupling factor to be examined
QL_vec  = Q0 / (beta_vec + 1)       # QL to be examined (derived from coupling factor)
dw_vec  = np.linspace(-50e3, 20e3, 8) * 2 * pi  # detuning to be examined, rad/s

# plot the required forward power and resulting reflected power for different 
# QL and detuning (in steady state)
status, Pfor, Pref = rf_power_req(f0, vc0, ib0, phib, Q0, roQ, 
                                  QL_vec       = QL_vec, 
                                  detuning_vec = dw_vec, 
                                  machine      = 'circular',
                                  plot         = True)

# calculate the optimal QL and detuning for minimizing the required power
status, QL_opt, dw_opt, beta_opt = opt_QL_detuning(f0, vc0, ib0, phib, Q0, roQ,
                                            machine  = 'circular',
                                            cav_type = 'nc')

print('Optimal beta = {:.3f}, QL = {:.3f}, detuning = {:.3f} Hz'.format(beta_opt, QL_opt, dw_opt/2/pi))

