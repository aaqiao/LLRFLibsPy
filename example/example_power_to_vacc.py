###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to estimate the accelerating voltage (steady-state) from RF drive 
power for:
    - standing-wave cavities
    - constant gradient traveling-wave structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np

from set_path import *
from rf_calib import *

# cavity voltage
roQ  = 1036                 # r/Q of the cavity (Linac convention), Ohm
QL   = 3e6                  # cavity loaded quality factor
pfor = 50e3                 # cavity drive RF power, W

status, vc0 = egain_cav_power(pfor, roQ, QL, machine = 'linac')

print('Cavity voltage for {:.3f} kW RF power is {:.3f} MV'.format(pfor/1000, vc0/1e6))

# traveling-wave structure voltage
L   = 4.15                  # length of the traveling-wave structure, m
rs  = 56.5e6                # shunt impedance per unit length, Ohm/m
Q   = 1.28e4                # quality factor of the TW strcuture
Tf  = 1e-6                  # filling time of the TW structure, s
f0  = 2998.8e6              # RF operating frequency, Hz
pwr = 24e6                  # input power to the structure, W

status, vacc0 = egain_cgstr_power(pwr, f0, rs, L, Q, Tf)

print('TW structure ACC voltage for {:.3f} MW RF power is {:.3f} MV'.format(pwr/1e6, vacc0/1e6))


