"""Simulate the RF cavity response in the presence of RF drive and beam loading."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for RF system simulator

Implemented:
    - cav_ss                : derive a continous state-space equation of a cavity (all modes)
    - cav_ss_passband       : derive a continous state-space equation of a cavity (only passband modes)
    - cav_ss_mech           : derive a continous state-space equation of mechanical modes
    - cav_impulse           : derive the cavity impulse response from the cavity parameters
    - sim_ncav_pulse        : simulate cavity (with constant QL and detuning) response to a pulsed input
    - sim_ncav_step         : simulate cavity (with constant QL and detuning) response for a time step
    - sim_ncav_step_simple  : simulate cavity (with constant QL and detuning) response for a time step
                              (simplified cavity equation only with the fundamental passband mode)
    - sim_scav_step         : simulate cavity response with mechanical modes for a time step
    - sim_ss_step           : a generic state-space solver to execute for one step
    - rf_power_req          : calculate the required RF power for diesired cavity voltage and beam
    - opt_QL_detuning       : calcualte the optimal QL and detuning for minimizing the reflection power

To be implemented:
    - implement the superconducting cavity model with Lorenz force detuning
    - traveling-wave structure model
    - pulse compressor (SLED/BOC) model
 
Some algorithms are referred to the following books:
S. Simrock and Z. Geng, Low-Level Radio Frequency Systems, Springer, 2022
https://link.springer.com/book/10.1007/978-3-030-94419-3 ("LLRF Book")
#########################################################################
'''
import numpy as np
from scipy import signal

from rf_sysid import *
from rf_misc import *

def cav_ss(half_bw, detuning = 0.0, beta = 1e4, passband_modes = None, 
           plot = False, plot_pno = 1000):
    '''
    Derive the continuous state-space equation of the cavity
     - include pass-band modes.
     - we assume the half-bandwidth and detuning of the fundamental mode are constant.
     - we output two models: one for RF drive, and the other for beam (beam only 
       interacts with the fundamental mode of the cavity).
       
    Refer to LLRF Book section 3.3.7 and 3.4.3.
    
    Parameters:
        half_bw:        float, half bandwidth of the cavity (constant), rad/s
        detuning:       float, detuning of the cavity (constant), rad/s
        beta:           float, input coupling factor (needed for NC cavities; 
                         for SC cavities, can use the default value, or you can 
                         specify it if more accurate result is needed)
        passband_modes: dict, with the following items:
                        ``freq_offs`` : list, offset frequencies of the modes, Hz;
                        ``gain_rel``  : list, relative gain wrt fundamental mode;
                        ``half_bw``   : list, half bandwidth of the mode, rad/s
        plot:           boolean, enable the plot of frequency response
        plot_pno:       int, number of point in the plot  
        
    Returns:
        status:             boolean, success (True) or fail (False)
        Arf, Brf, Crf, Drf: numpy matrix (complex), continous cavity model for RF drive
        Abm, Bbm, Cbm, Dbm: numpy matrix (complex), continous cavity model for beam drive
    '''
    # check the parameters
    if (half_bw <= 0) or (beta <= 0):
        return (False,) + (None,)*8

    if passband_modes is not None:
        if (not isinstance(passband_modes, dict)) or \
           ('freq_offs' not in passband_modes.keys()) or \
           ('gain_rel' not in passband_modes.keys()) or \
           ('half_bw' not in passband_modes.keys()): 
            return (False,) + (None,)*8

    plot_pno = 1000 if (plot_pno <= 0) else plot_pno

    # define the parameters of the fundamental mode
    b0 = 2 * beta * half_bw / (beta + 1)            # gain for the RF drive voltage
    b1 = 2 * half_bw                                # gain for the beam drive voltage

    # transfer function of the fundamental mode
    rf_num = [b0]
    rf_den = [1, half_bw - 1j*detuning]
    bm_num = [b1]
    bm_den = [1, half_bw - 1j*detuning]

    # interprete the passband modes
    if passband_modes is not None:
        # get the passband mode parameters
        pb_f  = passband_modes['freq_offs']         # frequency offset compared to the fundamental mode, Hz
        pb_g  = passband_modes['gain_rel']          # gain relative to the fundamental mode
        pb_wh = passband_modes['half_bw']           # half bandwidth of the passband modes, rad/s

        # add the passband mode transfer functions
        for i in range(len(pb_f)):
            rf_num, rf_den = add_tf(rf_num, rf_den, 
                                    [b0 * pb_g[i] * pb_wh[i] / half_bw], 
                                    [1, pb_wh[i] - 1j*2*np.pi*pb_f[i]])

    # get the state-space model
    Arf, Brf, Crf, Drf = signal.tf2ss(rf_num, rf_den)
    Abm, Bbm, Cbm, Dbm = signal.tf2ss(bm_num, bm_den)

    # plot the response
    if plot:
        # get the max freq range
        if passband_modes is not None:
            max_wrange = np.max([abs(detuning), 
                                 5*half_bw,
                                 np.max(np.abs(passband_modes['freq_offs']))*2*np.pi])
        else:
            max_wrange = np.max([abs(detuning), 
                                 5*half_bw])

        # calculate the response
        wrf, hrf = signal.freqs(rf_num, rf_den, worN = np.linspace(-2*max_wrange, 2*max_wrange, plot_pno))
        wbm, hbm = signal.freqs(bm_num, bm_den, worN = np.linspace(-2*max_wrange, 2*max_wrange, plot_pno))

        # make the plot
        from rf_plot import plot_cav_ss
        plot_cav_ss(wrf, hrf, wbm, hbm)

    # return the results
    return True, Arf, Brf, Crf, Drf, Abm, Bbm, Cbm, Dbm

def cav_ss_passband(passband_modes):
    '''
    Derive the continuous state-space equation of the cavity, only the passband modes.      
    Refer to LLRF Book section 3.3.7 and 3.4.3.
    
    Parameters:
        passband_modes: dict, with the following items:
                        ``freq_offs`` : list, offset frequencies of the modes, Hz;
                        ``gain_rel``  : list, relative gain wrt fundamental mode;
                        ``half_bw``   : list, half bandwidth of the mode, rad/s
    Returns:
        status:             boolean, success (True) or fail (False)
        Arf, Brf, Crf, Drf: numpy matrix (complex), continous passband model for RF drive
    '''
    # check the parameters
    if passband_modes is None:
        return (False,) + (None,)*4

    if (not isinstance(passband_modes, dict)) or \
       ('freq_offs' not in passband_modes.keys()) or \
       ('gain_rel'  not in passband_modes.keys()) or \
       ('half_bw'   not in passband_modes.keys()): 
        return (False,) + (None,)*4

    if len(passband_modes['freq_offs']) < 1:
        return (False,) + (None,)*4       

    # transfer function initialization
    rf_num = [0.0]
    rf_den = [1.0]

    # get the passband mode parameters
    pb_f  = passband_modes['freq_offs']         # frequency offset compared to the fundamental mode, Hz
    pb_g  = passband_modes['gain_rel']          # gain relative to the fundamental mode
    pb_wh = passband_modes['half_bw']           # half bandwidth of the passband modes, rad/s

    # add the passband mode transfer functions
    for i in range(len(pb_f)):
        rf_num, rf_den = add_tf(rf_num, rf_den, 
                                [pb_g[i] * pb_wh[i]], 
                                [1, pb_wh[i] - 1j*2*np.pi*pb_f[i]])

    # get the state-space model
    Arf, Brf, Crf, Drf = signal.tf2ss(rf_num, rf_den)

    # return the results
    return True, Arf, Brf, Crf, Drf

def cav_ss_mech(mech_modes, lpf_fc = None):
    '''
    Derive the continous state-space equation of the cavity mechanical modes.
    
    Refer to LLRF book section 3.3.10.
    
    Parameters:
        mech_modes: dict, with the following items:
            ``f`` : list, frequencies of mech modes, Hz;
            ``Q`` : list, qualify factors;
            ``K`` : list, K values, rad/s/(MV)^2.
        lpf_fc    : float, low-pass cutoff freq (optional), Hz
    Returns:
        status:     boolean, success (True) or fail (False)
        A, B, C, D: numpy matrix (real), continous mechanical model    
    '''
    # check 
    if mech_modes is None:
        return (False,) + (None,)*4
    
    if len(mech_modes['f']) < 1:
        return (False,) + (None,)*4
    
    # get the paremters
    fm = mech_modes['f']                # frequency of the mechanical mode, Hz
    Qm = mech_modes['Q']                # quality factor
    Km = mech_modes['K']                # K value, rad/s/(MV)^2    
    
    # build the transfer function of the first mechanical mode
    w = 2 * np.pi * fm[0]               # get the angular freq, rad/s
    num = [-Km[0] * w**2]
    den = [1, w/Qm[0], w**2]
        
    # add the other mode transfer functions
    if len(fm) > 1:
        for i in range(1, len(fm)):
            w = 2 * np.pi * fm[i]
            num, den = add_tf(num, den, [-Km[i] * w**2], [1, w/Qm[i], w**2])

    # add the LPF if applicable
    if lpf_fc is not None:
        if lpf_fc > 0.0:
            w = 2 * np.pi * lpf_fc
            num, den = add_tf(num, den, [w], [1, w])

    # get the state-space model
    A, B, C, D = signal.tf2ss(num, den)    
    
    # return the results
    return True, A, B, C, D

def cav_impulse(half_bw, detuning, Ts, order = 20):
    '''
    Derive the impulse response from the cavity equation. We assume that the 
    system gain has been normalized to 1 and the system phase corrected to 0. 
    Therefore, the referred cavity equation is ``dvc/dt + (half_bw - 1j * detuning)*vc = half_bw * vd``, 
    where ``vc`` is the vector of cavity voltage and ``vd`` is the vector of cavity 
    drive (in principle ``vd = 2 * beta * vf / (beta + 1)``).

    Refer to LLRF Book section 4.5.2.

    Parameters:
        half_bw:  float, constant half bandwidth of the cavity, rad/s
        detuning: float, constant detuning of the cavity, rad/s
        Ts:       float, sampling time, s
        order:    int, order of the impulse response    
        
    Returns:
        status:   boolean, success (True) or fail (False)
        h:        numpy array (complex), impulse response
    '''
    # check the input
    if (half_bw <= 0.0) or (detuning <= 0.0) or (Ts <= 0.0) or \
       (order < 2):
        return False, None

    # calculate the impulse response
    k = np.arange(order)
    h = Ts * half_bw * (1.0 - Ts * (half_bw - 1j * detuning))**k

    # return the results
    return True, h

def sim_ncav_pulse(Arfc, Brfc, Crfc, Drfc, vf, Ts, 
                   Abmc = None, 
                   Bbmc = None, 
                   Cbmc = None, 
                   Dbmc = None, 
                   vb   = None):
    '''
    Simulate the cavity response to a pulsed RF drive and beam current. This
    function is for normal conducting cavties with constant QL and detuning.

    Parameters:
        Arfc, Brfc, Crfc, Drfc: numpy matrix (complex), continous cavity model for RF drive
        vf:                     numpy array (complex), cavity forward voltage (calibrated to
                                 the cavity probe signal reference plane)
        Ts:                     float, sampling frequency, Hz
        Abmc, Bbmc, Cbmc, Dbmc: numpy matrix (complex), continous cavity model for beam drive
        vb:                     numpy array (complex), beam drive voltage (calibrated to
                                 the cavity probe signal reference plane)
                                 
    Returns:
        status: boolean, success (True) or fail (False)
        T:      numpy array, time waveform, s
        vc:     numpy array (complex), cavity voltage waveform
        vr:     numpy array (complex), cavity reflected voltage waveform
    '''
    # check the input
    if (Ts <= 0.0):
        return False, None, None, None

    if vb is not None:
        if (not vb.shape == vf.shape):
            return False, None, None, None

    # simulate the response of the continous system
    T = np.arange(vf.shape[0]) * Ts
    _, vc_rf, _ = signal.lsim((Arfc, Brfc, Crfc, Drfc), vf, T)      # Returns: T, Yout, Xout
    vc = vc_rf
    if not any([x is None for x in (Abmc, Bbmc, Cbmc, Dbmc, vb)]):
        _, vc_bm, _ = signal.lsim((Abmc, Bbmc, Cbmc, Dbmc), vb, T)
        vc += vc_bm

    # get the cavity reflected
    vr = vc - vf

    return True, T, vc, vr

def sim_ncav_step(Arfd, Brfd, Crfd, Drfd, vf_step, state_rf0,
                  Abmd      = None, 
                  Bbmd      = None, 
                  Cbmd      = None, 
                  Dbmd      = None, 
                  vb_step   = None,  
                  state_bm0 = None):
    '''
    Simulate the cavity response for a time step using the discrete cavity 
    state-space function.

    Parameters:
        Arfd, Brfd, Crfd, Drfd: numpy matrix (complex), discrete cavity model for RF drive
        vf_step:                complex, cavity forward voltage of this step
        state_rf0:              numpy matrix (complex), state of RF response model of last step
        Abmd, Bbmd, Cbmd, Dbmd: numpy matrix (complex), discrete cavity model for beam drive
        vb_step:                complex, beam drive voltage of this step
        state_bm0:              numpy matrix (complex), state of beam response model of last step
        
    Returns:
        status:   boolean, success (True) or fail (False)
        vc_step:  complex, cavity voltage of this step
        vr_step:  complex, cavity reflected voltage of this step
        state_rf: numpy matrix (complex), state of RF response model of this step (should input
                   to next execution)
        state_bm: numpy matrix (complex), state of beam response model of this step (should input
                   to next execution)
    '''
    # calculate the RF/beam response
    state_rf    = Arfd * state_rf0 + Brfd * vf_step
    vc_rf_step  = Crfd * state_rf0 + Drfd * vf_step

    if not any([x is None for x in (Abmd, Bbmd, Cbmd, Dbmd, vb_step, state_bm0)]):
        state_bm    = Abmd * state_bm0 + Bbmd * vb_step
        vc_bm_step  = Cbmd * state_bm0 + Dbmd * vb_step
        vc_step     = vc_rf_step[0,0] + vc_bm_step[0,0]
    else:
        state_bm    = None
        vc_step     = vc_rf_step[0,0]

    # get the cavity reflected of this step
    vr_step = vc_step - vf_step

    # return the results of the step
    return True, vc_step, vr_step, state_rf, state_bm

def sim_ncav_step_simple(half_bw, detuning, vf_step, vb_step, vc_step0, Ts, beta = 1e4):
    '''
    Simulate the cavity response for a time step using the simple discrete
    cavtiy equation (Euler method for discretization).

    Parameters:
        half_bw:  float, half bandwidth of the cavity (constant), rad/s
        detuning: float, detuning of the cavity (constant), rad/s
        vf_step:  complex, cavity forward voltage of this step
        vb_step:  complex, beam drive voltage of this step
        vc_step0: complex, cavity voltage of the last step
        Ts:       float, sampling time, s
        beta:     float, input coupling factor (needed for NC cavities; 
                   for SC cavities, can use the default value, or you can 
                   specify it if more accurate result is needed)   
    Returns:
        status:   boolean, success (True) or fail (False)
        vc_step:  complex, cavity voltage of this step
        vr_step:  complex, cavity reflected voltage of this step
    '''
    # check the input
    if (half_bw <= 0.0) or (Ts <= 0.0) or (beta <= 0.0):
        return False, None, None

    # make a step of calculation
    vc_step = (1 - Ts * (half_bw - 1j*detuning)) * vc_step0 + \
              2 * half_bw * Ts * (beta * vf_step / (beta + 1) + vb_step)
    vr_step = vc_step - vf_step

    # return the results of the step
    return True, vc_step, vr_step

def sim_scav_step(half_bw, dw_step0, detuning0, vf_step, vb_step, vc_step0, Ts, beta = 1e4,
                  state_m0 = 0, Am = None, Bm = None, Cm = None, Dm = None):
    '''
    Simulate the cavity response for a time step using the simple discrete
    cavtiy equation (Euler method for discretization) including the mechanical
    modes. We first simulate one step of the mechanical mode and determine the 
    detuning value, then use the ``sim_ncav_step_simple`` function to simulate
    one step of the electrical model.

    Parameters:
        half_bw:   float, half bandwidth of the cavity (constant), rad/s
        dw_step0:  float, detuning of the last step, rad/s
        detuning0: float, external detuning (tuner + microphonics), rad/s
        vf_step:   complex, cavity forward voltage of this step, V
        vb_step:   complex, beam drive voltage of this step, V
        vc_step0:  complex, cavity voltage of the last step, V
        Ts:        float, sampling time, s
        beta:      float, input coupling factor (needed for NC cavities; 
                    for SC cavities, can use the default value, or you can 
                    specify it if more accurate result is needed)   
        state_m0:  numpy matrix (real), last state of the mechanical equation 
        Am, Bm, Cm, Dm: numpy matrix (real), state-space matrix of mech modes
    Returns:
        status:   boolean, success (True) or fail (False)
        vc_step:  complex, cavity voltage of this step
        vr_step:  complex, cavity reflected voltage of this step
        dw:       float, detuning of this step, rad/s
        state_m:  numpy matrix (real), updated state of the mechanical equation
    '''
    # check the input
    if (half_bw <= 0.0) or (Ts <= 0.0) or (beta <= 0.0):
        return (False,) + (None,)*4

    # make a step of calculation of electrical equation (only pi mode)
    vc_step = (1 - Ts * (half_bw - 1j*dw_step0)) * vc_step0 + \
              2 * half_bw * Ts * (beta * vf_step / (beta + 1) + vb_step)
    vr_step = vc_step - vf_step

    # update the mechanical mode equation and get the detuning    
    if (state_m0 is None) or (Am is None) or (Bm is None) or (Cm is None) or (Dm is None):
        state_m = None
        dw      = detuning0
    else:
        state_m = Am * state_m0 + Bm * (abs(vc_step) * 1.0e-6)**2
        dw      = Cm * state_m0 + Dm * (abs(vc_step) * 1.0e-6)**2 + detuning0
       
    # return the results of the step
    return True, vc_step, vr_step, dw, state_m

def sim_ss_step(Ad, Bd, Cd, Dd, vin_step, state0):
    '''
    A generic state-space solver to execute for one step.
    Parameters:
        Ad, Bd, Cd, Dd: numpy matrix (float/complex), discrete state-space matrices
        vin_step:   float/complex, input of this step
        state0:     numpy matrix (float/complex), state of last step
    Returns:
        status:     boolean, success (True) or fail (False)
        vout_step:  float/complex, output of this step
        state:      numpy matrix (float/complex), state of this step (input to next exe)
    '''
    state     = Ad * state0 + Bd * vin_step
    vout_step = Cd * state0 + Dd * vin_step
    return True, vout_step, state

def rf_power_req(f0, vc0, ib0, phib, Q0, roQ_or_RoQ, 
                 QL_vec       = None,
                 detuning_vec = None, 
                 machine      = 'linac',
                 plot         = False):
    '''
    Plot the steady-state forward and reflected power for given cavity voltage, 
    beam current and beam phase, as function of loaded Q and detuning. The beam 
    phase is defined to be zero for on-crest acceleration.

    Refer to LLRF Book section 3.3.9.

    Parameters:
        f0:           float, RF operating frequency, Hz
        vc0:          float, desired cavity voltage, V
        ib0:          float, desired average beam current, A
        phib:         float, desired beam phase, degree
        Q0:           float, unloaded quality factor (for SC cavity, give it a
                       very high value like 1e10)
        roQ_or_RoQ:   float, cavity r/Q of Linac or R/Q of circular accelerator, Ohm
                       (see the note below)
        QL_vec:       numpy array, QL vector for power calculation
        detuning_vec: numpy array, detuning at which to evaluated the powers
        machine:      string, ``linac`` or ``circular``, used to select r/Q or R/Q
        plot:         boolean, enable/disable the plotting
        
    Returns:
        status:       boolean, success (True) or fail (False)
        Pfor:         dictionary, keyed by detuning, forward power at different QL
        Pref:         dictionary, keyed by detuning, reflected power at different QL

    Note: 
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines 
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to 
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (f0 <= 0.0) or (vc0 < 0.0) or (ib0 < 0.0) or (Q0 <= 0.0) or \
       (roQ_or_RoQ <= 0.0) or (QL_vec.shape[0] < 1):
        return False, None, None

    if detuning_vec is None:
        detuning_vec = [0.0]

    # calcualte the necessary parameters
    if machine == 'circular': RL_vec = roQ_or_RoQ * QL_vec          # calculate the loaded resistance RL, Ohm
    else:                     RL_vec = 0.5 * roQ_or_RoQ * QL_vec
    beta_vec = Q0 / QL_vec - 1.0                                    # input coupling factor
    wh_vec   = np.pi * f0 / QL_vec                                  # hald bandwidth, rad/s
    phib_rad = phib * np.pi / 180.0                                 # beam phase in radian

    # calculate for each detuning
    Pfor = {}
    Pref = {}
    for dw in detuning_vec:
        Pfor[dw] = (beta_vec + 1) / beta_vec * vc0**2 / 8 / RL_vec * (\
                   (1 + 2 * RL_vec * ib0 * np.cos(phib_rad) / vc0)**2 + \
                   (dw / wh_vec + 2 * RL_vec * ib0 * np.sin(phib_rad) / vc0)**2)
        Pref[dw] = (beta_vec + 1) / beta_vec * vc0**2 / 8 / RL_vec * (\
                   ((beta_vec - 1)/(beta_vec + 1) - 2 * RL_vec * ib0 * np.cos(phib_rad) / vc0)**2 + \
                   (dw / wh_vec + 2 * RL_vec * ib0 * np.sin(phib_rad) / vc0)**2)

    # plot the result
    if plot:
        from rf_plot import plot_rf_power_req
        plot_rf_power_req(Pfor, Pref, QL_vec)

    return True, Pfor, Pref

def opt_QL_detuning(f0, vc0, ib0, phib, Q0, roQ_or_RoQ, 
                    machine  = 'linac', 
                    cav_type = 'sc'):
    '''
    Derived the optimal loaded Q and detuning.

    Refer to LLRF Book section 3.3.9.

    Parameters:
        f0:           float, RF operating frequency, Hz
        vc0:          float, desired cavity voltage, V
        ib0:          float, desired average beam current, A
        phib:         float, desired beam phase, degree
        Q0:           float, unloaded quality factor (for SC cavity, 
                       give it a very high value like 1e10)
        roQ_or_RoQ:   float, cavity r/Q of Linac or R/Q of circular accelerator, Ohm
                       (see the note below)
        machine:      string, ``linac`` or ``circular``, used to select r/Q or R/Q
        cav_type:     string, ``sc`` for superconducting or ``nc`` for normal conducting
        
    Returns:
        status:       boolean, success (True) or fail (False)
        QL_opt:       float, optimal loaded quality factor
        dw_opt:       float, optimal detuning, rad/s
        beta_opt:     float, optimal input coupling factor

    Note: 
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines 
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to 
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (f0 <= 0.0) or (vc0 < 0.0) or (ib0 < 0.0) or (Q0 <= 0.0) or \
       (roQ_or_RoQ <= 0.0):
        return False, None, None

    # some parameters
    if machine == 'circular': shunt_imp = roQ_or_RoQ * 2.0
    else:                     shunt_imp = roQ_or_RoQ
    phib_rad = phib * np.pi / 180.0                   # beam phase in radian

    # calculate the optimal values
    dw_opt = -np.pi * f0 * shunt_imp * ib0 * np.sin(phib_rad) / vc0
    
    if cav_type == 'sc':
        QL_opt   = vc0 /(shunt_imp * ib0 * np.cos(phib_rad))
        beta_opt = Q0 / QL_opt - 1.0
    else:
        beta_opt = shunt_imp * Q0 * ib0 * np.cos(phib_rad) / vc0 + 1.0
        QL_opt   = Q0 / (beta_opt + 1.0)

    # return the results
    return True, QL_opt, dw_opt, beta_opt





















