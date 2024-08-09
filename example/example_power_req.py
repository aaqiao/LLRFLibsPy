###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to estimate the required RF power (steady-state) for specific beam
loading and accelerating voltage/phase for standing-wave cavities 
This example is for a superconducting cavity in Linac
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *

# parameters
pi      = np.pi                         # shorter pi
f0      = 1.3e9                         # RF operating frequency, Hz
vc0     = 25e6                          # desired cavity voltage, V
ib0     = 0.008                         # average beam current, A
phib    = -30.0                         # desired beam phase (Linac convention), deg
Q0      = 1e10                          # cavity unloaded quality factor
roQ     = 1036                          # cavity r/Q (Linac convention), Ohm
QL_vec  = np.linspace(1e6, 1e7, 100)    # loaded quality factor evaluated
dw_vec  = np.linspace(0, 2*pi*200, 5)   # detuning evaluated

# plot the required forward power and resulting reflected power for different 
# QL and detuning (in steady state)
status, Pfor, Pref = rf_power_req(f0, vc0, ib0, phib, Q0, roQ, 
                                  QL_vec       = QL_vec, 
                                  detuning_vec = dw_vec, 
                                  plot         = True)

# calculate the optimal QL and detuning for minimizing the required power
status, QL_opt, dw_opt, _ = opt_QL_detuning(f0, vc0, ib0, phib, Q0, roQ)

print('Optimal QL = {:.3f}, detuning = {:.3f} Hz'.format(QL_opt, dw_opt/2/pi))



