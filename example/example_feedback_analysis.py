###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to analyze the stability margins of an RF feedback loop based on
a basic controller (PI + notches) and a cavity simulator
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
f_scale = 6                         # sampling frequency scaling factor from the default 1 MHz
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

wh      = pi*f0 / QL                # half-bandwidth of the cavity, rad/s
dw      = wh                        # detuning of the cavity, rad/s

pb_modes = {'freq_offs': [-800e3],              # offset frequencies of cavity passband modes, Hz
            'gain_rel':  [-1],                  # gain of passband modes compared to the pi-mode
            'half_bw':   [2*np.pi*216 * 0.5]}   # half-bandwidth of the passband modes, rad/s

# ---------------------------------------------------------
# define the cavity model
# ---------------------------------------------------------
# cavity model (change plot to True for showing the frequency response)
result = cav_ss(wh, detuning = dw, passband_modes = pb_modes, plot = True)
Arf    = result[1]                  # A,B,C,D of cavity model for RF drive
Brf    = result[2]
Crf    = result[3]
Drf    = result[4]

# discretize the cavity equation (change plot to True to plot freq. response)
status1, Arfd, Brfd, Crfd, Drfd, _ = ss_discrete(Arf, Brf, Crf, Drf, Ts, 
                                              method = 'zoh', 
                                              plot   = True)

# create a notch filter for removing the passband mode from measurement
# Note: this notch filter is only applied to the cavity output measurement to
#       avoid the instability caused by passband modes. It should be distingished
#       from the notch filter in the feedback controller (see below)
status, Afd, Bfd, Cfd, Dfd = design_notch_filter(800e3, 4, fs)
ss_freqresp(Afd, Bfd, Cfd, Dfd, Ts = Ts, title = 'Notch Filter Discrete', plot = True)

# cascade the cavity model and the notch filter to form the plant to be controlled
status, AGd, BGd, CGd, DGd, _ = ss_cascade(Arfd, Brfd, Crfd, Drfd, Afd, Bfd, Cfd, Dfd, Ts = Ts)
ss_freqresp(AGd, BGd, CGd, DGd, Ts = Ts, title = 'Cascaded Discrete', plot = True)

# ---------------------------------------------------------
# define the general feedback controller
# ---------------------------------------------------------
Kp = 30                                 # proportional gain
Ki = 1e5                                # integral gain

notches = {'freq_offs': [5e3],          # notch filter config for FB
           'gain':      [1000],
           'half_bw':   [2*np.pi*20]}

# basic RF controller (set plot to True to show the frequency response)
status, Akc, Bkc, Ckc, Dkc = basic_rf_controller(Kp, Ki,
                                                 notch_conf = notches,      # set to None to see results without notch filters
                                                 plot       = True, 
                                                 plot_maxf  = 10e3)

# get the discrete controller (set plot to True to show the freq. responses)
status, Akd, Bkd, Ckd, Dkd, _ = ss_discrete(Akc, Bkc, Ckc, Dkc, Ts, 
                                         method     = 'bilinear', 
                                         plot       = True,
                                         plot_pno   = 10000)

# ---------------------------------------------------------
# calculate the loop properties: open-loop response, sensitivity, complementary sensitivity
# ---------------------------------------------------------
delay_s = 0                             # delay will affect the stability and margins
#delay_s = 1.9e-6                       # adjust the delay to mitigate the passband modes caused instability

rc  = loop_analysis(Arf, Brf, Crf, Drf, 
                    Akc, Bkc, Ckc, Dkc, 
                    delay_s = delay_s, 
                    label   = 'Continous without Notch Filter')
rd  = loop_analysis(AGd, BGd, CGd, DGd, 
                    Akd, Bkd, Ckd, Dkd, 
                    Ts      = Ts, 
                    delay_s = delay_s,
                    label   = 'Discrete with Notch Filter')
rd2 = loop_analysis(Arfd, Brfd, Crfd, Drfd, 
                    Akd,  Bkd,  Ckd,  Dkd, 
                    Ts      = Ts, 
                    delay_s = delay_s,
                    label   = 'Discrete without Notch Filter')


















