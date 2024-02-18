"""Design and analyze RF feedback/feedforward controllers."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for RF controller design, analysis and simulation

Implemented:
    - ss_discrete         : discretize a continous state-space system and compare freq responses
    - ss_cascade          : cascade two state-space systems (either continous or discrete - C/D)
    - ss_freqresp         : calculate and plot freq response of a state-space system (C/D)
    - basic_rf_controller : derive a basic continous RF I/Q controller: P + I + frequency notches
    - control_step        : perform one time-step execution of the discretized controller
    - loop_analysis       : analyze the sensitivity/complementary sensitivity of an RF control loop (C/D)
    - cav_sp_ff           : derive the setpoint and feedforward waveforms for desired cavity voltage
                            and beam loading
    - ADRC_controller     : derive a basic ADRC controller (the observer and gain)
    - ADRC_control_step   : perform one time-step execution of the discretized controller including
                            the ADRC observer
    - AFF_timerev_lpf     : time-reversed low pass filter-based adaptive feedforward
    - AFF_ilc_design      : derive the ILC gain matrix from the impulse response and weighting
    - AFF_ilc             : apply the ILC algorithm to calculate the feedforward correction signal
    - resp_inv_svd        : response matrix inversion with SVD (with singular value filtering)
    - resp_inv_lsm        : response matrix inversion with lease-square method (with regularization)

To be implemented:
    - in "cav_sp_ff", smooth the feedforward and update the setpoint correspondingly
    - pulsed piezo drive generation
    - Active noise cancellation for tuning control (narro-band/wideband ANC)
    - look-up table based operating point determination
 
Some algorithms are referred to the following books:
S. Simrock and Z. Geng, Low-Level Radio Frequency Systems, Springer, 2022
https://link.springer.com/book/10.1007/978-3-030-94419-3 ("LLRF Book")
Z. Geng and S. Simrock, Intelligent Beam Control in Accelerators, Springer, 2023
https://link.springer.com/book/10.1007/978-3-031-28597-4 ("Beam Control Book")
#########################################################################
'''
import numpy as np
from scipy import signal

from rf_sysid import *
from rf_misc import *

def ss_discrete(Ac, Bc, Cc, Dc, Ts, method = 'zoh', alpha = 0.3, plot = False, plot_pno = 1000):
    '''
    Derive the discrete state-space equation from a continous one and compare 
    their frequency responses.
    
    Parameters:
        Ac, Bc, Cc, Dc: numpy matrix (complex), continous state-space model
        Ts:             float, sampling time, s
        method:         string, ``gbt``, ``bilinear``, ``euler``, ``backward_diff``, ``zoh``, 
                         ``foh`` or ``impulse`` (see document of signal.cont2discrete)
        alpha:          float, 0 to 1 (see document of signal.cont2discrete) 
        plot:           boolean, enable the plot of frequency responses
        plot_pno:       int, number of point in the plot
        
    Returns:
        status:         boolean, success (True) or fail (False)
        Ad, Bd, Cd, Dd: numpy matrix (complex), discrete state-space model
    '''
    # check the input
    if (Ts <= 0):
        return (False,) + (None,)*4

    plot_pno = 1000 if (plot_pno <= 0) else plot_pno

    # discretize it
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), Ts, method = method, alpha = alpha)

    # plot the responses
    if plot:
        # calculate the responses of both continous and discrete versions
        sc, fc, Ac_dB, Pc_deg,_ = ss_freqresp(Ac, Bc, Cc, Dc, plot_pno = plot_pno, plot_maxf = 1.0 / Ts)
        sd, fd, Ad_dB, Pd_deg,_ = ss_freqresp(Ad, Bd, Cd, Dd, plot_pno = plot_pno, Ts = Ts)

        # make the plot
        if sc and sd:
            from rf_plot import plot_ss_discrete
            plot_ss_discrete(fc, Ac_dB, Pc_deg, fd, Ad_dB, Pd_deg, Ts)

    # return the results
    return True, Ad, Bd, Cd, Dd

def ss_cascade(A1, B1, C1, D1, A2, B2, C2, D2, Ts = None):
    '''
    Cascade two state-space systems with the first system applyed to
    the input first. This function works for both continous system (``Ts`` is None)
    and discrete systems (``Ts`` has a nonzero floating value).
    
    Parameters:
        A1, B1, C1, D1: numpy matrix (complex), state-space model of system 1
        A2, B2, C2, D2: numpy matrix (complex), state-space model of system 2
        Ts:             float, sampling time, s
        
    Returns:
        status          boolean, success (True) or fail (False)
        A, B, C, D:     numpy matrix (complex), state-space model of cascaded system
    '''
    # check the input
    if Ts is not None:
        if Ts <= 0.0:
            return (False,) + (None,)*5

    # construct the state-space object
    if Ts is None:                      # continuous
        sys1 = signal.StateSpace(A1, B1, C1, D1)
        sys2 = signal.StateSpace(A2, B2, C2, D2)
    else:                               # discrete
        sys1 = signal.StateSpace(A1, B1, C1, D1, dt = Ts)
        sys2 = signal.StateSpace(A2, B2, C2, D2, dt = Ts)

    # cascade the system
    sys  = sys2 * sys1

    # return the results
    return True, sys.A, sys.B, sys.C, sys.D, Ts

def ss_freqresp(A, B, C, D, Ts = None, plot = False, plot_pno = 1000, plot_maxf = 0.0, 
                title = 'Frequency Response'):
    '''
    Plot the frequency response of a state-space system. This function works 
    for both continous system (``Ts`` is None) and discrete systems (``Ts`` has a 
    nonzero floating value).
    
    Parameters:
        A, B, C, D: numpy matrix (complex), state-space model of system
        Ts:         float, sampling time, s
        plot:       boolean, enable the plot of frequency response
        plot_pno:   int, number of point in the plot
        plot_maxf:  float, frequency range (+-) to be plotted, Hz
        title:      string, title showed on the plot
        
    Returns:
        status:     boolean, success (True) or fail (False)
        f_wf:       numpy array, frequency waveform, Hz
        A_wf_dB:    numpy array, amplitude response waveform, dB
        P_wf_deg:   numpy array, phase response waveform, deg
        h:          numpy array (complex), complex response
    '''    
    # check the input
    plot_pno  = 1000 if (plot_pno  <= 0) else plot_pno
    plot_maxf = 1e6  if (plot_maxf <= 0) else plot_maxf

    if Ts is not None:
        if Ts <= 0.0:
            return (False,) + (None,)*4

    # calculate the frequency response 
    if Ts is None:                      # continous
        maxw = 2 * np.pi * plot_maxf
        fs   = 1.0
        w, h = signal.freqresp((A, B, C, D), w = np.linspace(-maxw, maxw, plot_pno))
    else:                               # discrete
        maxw = np.pi
        fs   = 1.0 / Ts
        w, h = signal.dfreqresp((A, B, C, D, Ts), w = np.linspace(-maxw, maxw, plot_pno))

    # calculate the results for display
    f_wf     = w / 2 / np.pi * fs
    A_wf_dB  = 20 * np.log10(np.abs(h))
    P_wf_deg = np.angle(h, deg = True)

    # plot 
    if plot:
        from rf_plot import plot_ss_freqresp
        plot_ss_freqresp(f_wf, A_wf_dB, P_wf_deg, h, fs, Ts, title)

    # return the frequency response (frequency in absolute Hz)
    return True, f_wf, A_wf_dB, P_wf_deg, h

def basic_rf_controller(Kp, Ki, notch_conf = None, plot = False, plot_pno = 1000, plot_maxf = 0.0):
    '''
    Generate the continous state-space equation for a basic controller with
     * I/Q control strategy (input/output are all complex signals).
     * configurable for PI control + frequency notches.
    
    Parameters:
        Kp:          float, proportional (P) gain
        Ki:          float, integral (I) gain
        notch_conf:  dict, with the following items:
            ``freq_offs``: list, offset frequency to be notched, Hz; 
            ``gain``: list, gain of the notch (inverse of suppress ratio); 
            ``half_bw``: list, half bandwidth of the notch filter, rad/s
        plot:        boolean, enable the plot of frequency response of the controller
        plot_pno:    int, number of point in the plot
        plot_maxf:   float, frequency range (+-) to be plotted, Hz   
        
    Returns:
        status:             boolean, success (True) or fail (False)
        Akc, Bkc, Ckc, Dkc: numpy matrix (complex), continuous state-space controller
    '''
    # check the input
    if (Kp < 0) or (Ki < 0):
        return (False,) + (None,)*4

    if notch_conf is not None:
        if (not isinstance(notch_conf, dict)) or \
           ('freq_offs' not in notch_conf.keys()) or \
           ('gain'      not in notch_conf.keys()) or \
           ('half_bw'   not in notch_conf.keys()): 
            return (False,) + (None,)*4

    plot_pno  = 1000 if (plot_pno  <= 0) else plot_pno
    plot_maxf = 1e6  if (plot_maxf <= 0) else plot_maxf

    # first get the PI controller
    K_num, K_den = add_tf([Kp], [1.0], [Ki], [1.0, 0.0])

    # add the notch filters
    if notch_conf is not None:
        # get the notch parameters
        nt_f  = notch_conf['freq_offs']     # frequency offset, Hz
        nt_g  = notch_conf['gain']          # gain
        nt_wh = notch_conf['half_bw']       # half bandwidth of the notch filter, rad/s

        # add the notch filter transfer functions
        for i in range(len(nt_f)):
            K_num, K_den = add_tf(K_num, K_den, 
                                  [nt_g[i] * nt_wh[i]], 
                                  [1.0, nt_wh[i] - 1j*2*np.pi*nt_f[i]])
            K_num, K_den = add_tf(K_num, K_den, 
                                  [nt_g[i] * nt_wh[i]], 
                                  [1.0, nt_wh[i] + 1j*2*np.pi*nt_f[i]])

    # get the state-space model
    Akc, Bkc, Ckc, Dkc = signal.tf2ss(K_num, K_den)

    # plot the response
    if plot:
        # calculate the response
        w, h = signal.freqs(K_num, K_den, 
                            worN = np.linspace(-2*np.pi*plot_maxf, 
                                                2*np.pi*plot_maxf, 
                                                plot_pno))

        # make the plot
        from rf_plot import plot_basic_rf_controller
        plot_basic_rf_controller(w, h)

    # return the results
    return True, Akc, Bkc, Ckc, Dkc

def control_step(Akd, Bkd, Ckd, Dkd, err_step, state_k0, ff_step = 0.0):
    '''
    Controller execute for one step based on the discrete state-space equation.
    
    Parameters:
        Akd, Bkd, Ckd, Dkd: numpy matrix (complex), discrete state-space controller
        err_step:           complex, system output error (input to controller) of this step
        state_k0:           numpy matrix (complex), state of the last step
        ff_step:            complex, feedforward of this step
        
    Returns:
        status:             boolean, success (True) or fail (False)
        ctrl_step:          complex, overall output of the controller of this step
        ctrl_out:           complex, feedback output (exclude feedforward wrt ctrl_step)
        state_k:            numpy matrix (complex), state of this step (should be input to 
                             the next execution of this function)
        
    Question: 
        In the second equation, shall we use ``state_k0`` or ``state_k``?
    '''
    # calculate the controller output
    state_k   = Akd * state_k0 + Bkd * err_step
    ctrl_out  = Ckd * state_k0 + Dkd * err_step
    ctrl_step = ctrl_out + ff_step

    # return the results of the step
    return True, ctrl_step, ctrl_out, state_k

def loop_analysis(AG, BG, CG, DG, AK, BK, CK, DK, Ts = None, delay_s = 0, 
                  plot = True, plot_pno = 100000, plot_maxf = 0.0, label = ''):
    '''
    Control loop analysis, including
     * derive the open loop transfer function.
     * calculate the senstivity and complementary sensitivity functions. 
     
    This function works for both continous system (``Ts`` is None) and discrete 
    systems (``Ts`` has a nonzero floating value).
    
    Parameters:
        AG, BG, CG, DG: numpy matrix (complex), plant model
        AK, BK, CK, DK: numpy matrix (complex), controller
        Ts:             float, sampling time, s
        delay_s:        float, loop delay, s        
        plot:           boolean, enable the plot of bode and Nyquist plots
        plot_pno:       int, number of point in the plot
        plot_maxf:      float, frequency range (+-) to be plotted, Hz    
        
    Returns:
        status:         boolean, success (True) or fail (False)
        S_max:          float, maximum sensitivity, dB
        T_max:          float, maximum complementary sensitivity, dB
    '''
    # check the input
    plot_pno  = 100000 if (plot_pno  <= 0) else plot_pno
    plot_maxf = 1e6    if (plot_maxf <= 0) else plot_maxf
    delay_s   = 0.0    if (delay_s   <  0) else delay_s

    if Ts is not None:
        if Ts <= 0.0:
            return False, None, None

    # parameter based on continous or discrete (Ts not None)
    fs = 1.0 if (Ts is None) else 1.0/Ts

    # cascade the plant and the controller get the open loop L
    status, AL, BL, CL, DL, _ = ss_cascade(AK, BK, CK, DK, AG, BG, CG, DG, Ts = Ts)
    if not status:
        return False, None, None

    # frequency response of L
    result_L    = ss_freqresp(AL, BL, CL, DL, Ts = Ts, 
                              plot_pno = plot_pno, plot_maxf = plot_maxf)
    status      = result_L[0]
    f_wf        = result_L[1]
    AL_dB       = result_L[2]
    L           = result_L[4]                       # complex response of L

    if not status:
        return False, None, None

    # apply the delay to the frequency response of L
    L *= np.exp(-1j * 2.0 * np.pi * f_wf * delay_s)

    # sensitivity and complementary sensitivity
    S = 1.0 / (1.0 + L)
    T = 1.0 - S

    # plot (similar to ss_freqresp)
    if plot:
        from rf_plot import plot_loop_analysis
        plot_loop_analysis(f_wf, L, S, T, fs, Ts, label)

    # calculate the peak of S and T
    S_max = np.max(20*np.log10(np.abs(S)))
    T_max = np.max(20*np.log10(np.abs(T)))

    return True, S_max, T_max

def cav_sp_ff(half_bw, filling_len, flattop_len, Ts, pno,
                vc0        = 1.0,
                detuning   = 0.0, 
                beta       = 1e4, 
                const_fpow = True,
                ib0        = None,
                phib_deg   = 0.0,
                beam_ids   = 0,
                beam_ide   = 0,
                roQ_or_RoQ = 0.0,
                QL         = 3e6,
                machine    = 'linac'):
    '''
    Generate the basic cavity set point and feedforward used to configure
    the LLRF controller (later may apply smooth in the edges).
    
    Refer to LLRF Book section 9.2.1.
    
    Parameters:
        half_bw:     float, half bandwidth of the cavity, rad/s
        filling_len: int, length of cavity filling time, number of samples
        flattop_len: int, length of the flattop for beam acc., number of samples
        Ts:          float, sampling time, s
        pno:         int, number of samples in the returned waveforms
        vc0:         float, desired cavity voltage at the flattop, V
        detuning:    float, detuning of the cavity, rad/s
        beta:        float, input coupling factor (needed for NC cavities; 
                      for SC cavities, can use the default value, or you can 
                      specify it if more accurate result is needed)
        const_fpow:  boolean, True for forcing constant filling drive power/phase
        ib0:         float, average beam current, A
        phib_deg:    float, beam accelerating phase (0 for on-crest), deg
        beam_ids:    int, beam starting sample index
        beam_ide:    int, beam ending sample index
        roQ_or_RoQ:  r/Q of Linac or R/Q of circular accelerator, Ohm
                      (see the note below)
        QL:          float, loaded quality factor of the cavity
        machine:     string, ``linac`` or ``circular``, used to select r/Q or R/Q
        
    Returns:
        status:      boolean, success (True) or fail (False)
        sp:          numpy array (complex), set point waveform (for controller)
        vf_ff:       numpy array (complex), feedforward waveform (for controller)
        vb:          numpy array (complex), beam drive voltage waveform
        Tpul:        numpy array, time array for the waveforms
    
    Note: 
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines 
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to 
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (half_bw <= 0) or (filling_len <= 0) or (flattop_len <= 0) or \
       (Ts <= 0) or (pno < filling_len + flattop_len):
        return (False,) + (None,)*4

    vc0  = 1.0 if (vc0  <= 0.0) else vc0
    beta = 1e4 if (beta <= 0.0) else beta
    
    if ib0 is not None:
        if (ib0 < 0.0) or (beam_ids < 0) or (beam_ide <= beam_ids) or \
           (roQ_or_RoQ <= 0.0) or (QL <= 0.0):
            return (False,) + (None,)*4

    # define the setpoint table
    N  = filling_len + flattop_len          # effective pulse width
    T  = np.arange(filling_len+1) * Ts
    sp = np.ones(N, dtype = complex) 
    sp[:filling_len+1] = (1.0 - np.exp(-half_bw * T)) / (1.0 - np.exp(-half_bw * T[-1]))
    sp *= vc0
    
    # derive the ff of vf
    status, vf_ff, _ = cav_drv_est(sp, half_bw, Ts, detuning, beta)
    if not status:
        return (False,) + (None,)*4

    # handle the filling power if require to be constant (see Example 3.1 of LLRF Book)
    if const_fpow:
        vf_fill = vc0 * (half_bw - 1j*detuning) / \
                  (1.0 - np.exp(-(half_bw - 1j*detuning) * T[-1])) * \
                  (beta + 1) / (2 * half_bw * beta)
        vf_ff[:filling_len+1] = vf_fill

    # calculate the beam drive voltage
    vb = np.zeros(vf_ff.shape, dtype = complex)
    if ib0 is not None:
        # calculate the loaded resistance
        if machine == 'circular':
            RL = roQ_or_RoQ * QL
        else:
            RL = 0.5 * roQ_or_RoQ * QL
       
        # determine the beam drive
        vb[beam_ids:beam_ide+1] = RL * ib0 * np.exp(1j*(np.pi - phib_deg*np.pi/180.0))

    # remove the beam term
    vf_ff -= (beta + 1) / beta * vb

    # attend the pulse to the desired length
    if pno > N:
        sp    = np.hstack((sp,    np.zeros(pno - N)))
        vf_ff = np.hstack((vf_ff, np.zeros(pno - N)))
        vb    = np.hstack((vb,    np.zeros(pno - N)))

    return True, sp, vf_ff, vb, np.arange(pno)*Ts

def ADRC_controller(half_bw, pole_scale = 50.0):
    '''
    Generate the continous ADRC controller. We assume that the system gain
    has been normalized to 1 and the system phase corrected to 0. Therefore,
    the referred cavity equation is
    ``dvc/dt + (half_bw - 1j * detuning)*vc = half_bw * vd``
    where ``vc`` is the vector of cavity voltage and ``vd`` is the vector of cavity 
    drive (in principle ``vd = 2 * beta * vf / (beta + 1)``).
    
    Refer to LLRF Book section 4.2.3.
    
    Parameters:
        half_bw:    float, half bandwidth of the cavity, rad/s
        pole_scale: float, define the pole location of the observer
        
    Returns:
        status:                 boolean, success (True) or fail (False)
        Aobc, Bobc, Cobc, Dobc: numpy matrix (complex), continous ADRC observer
        b0:                     float, ADRC gain
    '''
    # check the input
    if (half_bw <= 0) or (pole_scale <= 0):
        return (False,) + (None,)*5

    # parameters
    p_obs = -pole_scale * half_bw           # pole of the observer
    l1    = -2 * p_obs                      # observer matrix parameter
    l2    = p_obs**2                        # observer matrix parameter
    b0    = half_bw                         # gain for RF drive voltage

    # construct the continous ADRC observer
    Aobc = np.matrix([[-l1, 1],
                      [-l2, 0]], dtype = complex)
    Bobc = np.matrix([[ l1, b0],
                      [ l2,  0]], dtype = complex)
    Cobc = Dobc = np.matrix(np.zeros(Aobc.shape), dtype = complex)

    # return the ADRC observer and gain
    return True, Aobc, Bobc, Cobc, Dobc, b0

def ADRC_control_step(Akd, Bkd, Ckd, Dkd, 
                      Aobd, Bobd, b0, 
                      sp_step, vc_step, vd_step,
                      state_k0, state_ob0, 
                      vf_step = 0.0, 
                      ff_step = 0.0, 
                      apply_to_err = False):
    '''
    Controller execute for one step based on the controller's discrete state-space equation
    including the ADRC observer.
    
    Refer to LLRF Book Figure 4.6.
    
    Parameters:
        Akd, Bkd, Ckd, Dkd: numpy matrix (complex), discrete RF controller
        Aobd, Bobd:         numpy matrix (complex), discrete ADRC observer (C,D not needed)
        b0:                 float, ADRC gain
        sp_step:            complex, cavity voltage setpoint of this time step
        vc_step:            complex, cavity voltage meas. of last time step
        vd_step:            complex, cavity drive of last time step
        vf_step:            complex, feedfoward part of cavity drive of last time step
        state_k0:           numpy matrix (complex), controller state of last time step
        state_ob0:          numpy matrix (complex), ADRC observer state of last time step
        ff_step:            complex, feedforward of this time step
        apply_to_err:       boolean, True to apply ADRC to error, or apply to whole cavity voltage
        
    Returns:
        status:             boolean, success (True) or fail (False)
        ctrl_step:          complex, overall output of the controller of this step
        ctrl_out:           complex, feedback output (exclude feedforward wrt ctrl_step)
        state_k:            numpy matrix (complex), controller state of this step (should input to 
                             the next execution of this function)
        state_ob:           numpy matrix (complex), ADRC observer state of this step (should input
                             to the next execution of this function)
        f:                  complex, estimated general disturbance by the ADRC observer
        
    Question: 
        In the second equation, shall we use ``state_k0`` or ``state_k``?
    
    Note: 
          1. looks like ADRC can mitigate the instability caused by the cavity's
             passband modes, then no notch filter is required to filter the cavity 
             voltage measurement.
          2. basic ADRC is perfect with a proportional controller - resulting
             in zero steady-state error. Looks like with other controller (e.g.,
             PI control), ADRC needs to be revised for good performance.
          3. adding feedforward to the basic ADRC also causes some problem, like
             the steady state error becomes nonzero.
          4. The solution to handle the problems of point 2 and 3 above is to apply
             ADRC to the errors of the cavity drive and voltage. See the paper:
             S. Zhao et al, Tracking and disturbance rejection in non-minimum
             phase systems, Proceedings of the 33rd Chinese Control Conference, 
             pp. 3834-3839, July 28-30, 2014, Nanjing, China.
    '''
    # execute one step based on user preference
    if apply_to_err:
        # execute observer and estimate cavity voltage error (vc_err_est) and disturbance (f)
        state_ob = Aobd * state_ob0 + Bobd * np.matrix([[vc_step - sp_step], [vd_step - vf_step]])
        vc_err_est, f = -state_ob[0, 0], state_ob[1, 0]

        # calculate the controller output
        state_k  = Akd * state_k0 + Bkd * vc_err_est
        ctrl_out = Ckd * state_k0 + Dkd * vc_err_est

        # calculate the final drive to the cavity (slightly different from Fig.4.6, 
        # we do not divide controller output by b0)
        ctrl_step = ctrl_out - f / b0 + ff_step
        vc_est    = sp_step - vc_err_est

    else:
        # execute observer and estimate cavity voltage (vc_est) and disturbance (f)
        state_ob = Aobd * state_ob0 + Bobd * np.matrix([[vc_step], [vd_step]])
        vc_est, f = state_ob[0, 0], state_ob[1, 0]

        # calculate the controller output
        state_k  = Akd * state_k0 + Bkd * (sp_step - vc_est)
        ctrl_out = Ckd * state_k0 + Dkd * (sp_step - vc_est)

        # calculate the final drive to the cavity (slightly different from Fig.4.6, 
        # we do not divide controller output by b0)
        ctrl_step = ctrl_out - f / b0 + ff_step

    # return the results of the step
    return True, ctrl_step, ctrl_out, state_k, state_ob, vc_est, f

def AFF_timerev_lpf(vfb, fcut, fs, vff_cor = None):
    '''
    Time-reversed low-pass filter, we only apply the first order IIR low-pass,
    which gives up to 90 degrees phase lead.
    
    Refer to LLRF Book section 4.5.1.

    Parameters:
        vfb:     numpy array (complex), feedback control waveform
        fcut:    float, cut-off frequency of the low-pass filter, Hz
        fs:      float, sampling frequency, Hz
        vff_cor: numpy array (complex), buffer storing the filtered waveform
        
    Returns:
        status:  boolean, success (True) or fail (False)
        vff_cor: numpy array (complex), feedforward correction waveform
    '''
    # check the input
    if (vfb.shape[0] < 3) or (fcut <= 0.0) or \
       (fcut >= fs/2) or (fs <= 0.0):
        return False, None

    if vff_cor is not None:
        if (not vfb.shape == vff_cor.shape):
            return false, None

    # perform the time-reversed filtering 
    vfb = vfb[::-1]                         # reverse the time of input
    a   = 2.0 * np.pi * fcut / fs           # scale factor
    if vff_cor is None:
        vff_cor = np.zeros(vfb.shape, dtype = complex) # create buffer

    for i in range(1, vfb.shape[0]):        # filtering
        vff_cor[i] = (1.0 - a) * vff_cor[i-1] + a * vfb[i]

    # return the result
    return True, vff_cor[::-1]
    
def AFF_ilc_design(h, pulw, P = None, Q = None):
    '''
    Adaptive feedforward with optimal iterative learning control (ILC).
    
    Refer to LLRF Book section 4.5.2.

    Parameters:
        h:      numpy array (complex), impulse response
        pulw:   int, pulse width as number of points
        P, Q:   numpy matrix, positive-definite weight matrices
        
    Returns:
        status: boolean, success (True) or fail (False)
        L:      numpy matrix (complex), gain matrix of ILC
    '''
    # check the input
    if (h.shape[0] < 3) or (pulw < 3):
        return False, None

    if P is None:
        P = np.matrix(np.eye(pulw))
    if Q is None:
        Q = np.matrix(np.eye(pulw))

    if (not P.shape == Q.shape) or (not P.shape[0] == pulw):
        return False, None

    # dimension
    order = h.shape[0]

    # derive the system transfer matrix
    G = np.zeros((pulw, pulw), dtype = complex)
    for i in range(pulw):
        G[i:, i] = h[:min(pulw-i, order)]

    # calculate the ILC gain matrix
    G = np.matrix(G)
    L = np.linalg.inv(Q + G.H * P * G) * G.H * P

    return True, L

def AFF_ilc(vc_err, L):
    '''
    Apply the ILC.

    Refer to LLRF Book section 4.5.2.
    
    Parameters:
        vc_err:  numpy array (complex), error of the cavity voltage waveform
        L:       numpy matrix (complex), gain matrix of ILC
        
    Returns:
        vff_cor: numpy array (complex), feedforward correction waveform
    '''
    return np.matmul(L, vc_err)

def resp_inv_svd(R, singular_val_filt = 0.0):
    '''
    Response matrix inversion with SVD.

    Refer to Beam Control Book section 2.4.2.
    
    Parameters:
        R:                 numpy matrix, response matrix
        singular_val_filt: float, threshold of singular values, the ones 
                            smaller or equal to it will be discarded
                            
    Returns:
        Rinv: numpy matrix, inversion of the response matrix
    '''
    return np.linalg.pinv(R, rcond = singular_val_filt)

def resp_inv_lsm(R, regu = 0.0):
    '''
    Response matrix inversion with least-square method.
    
    Refer to Beam Control Book section 2.4.3.
    
    Parameters:
        R:    numpy matrix, response matrix
        regu: float, regularization factor (should > 0)
        
    Returns:
        Rinv: numpy matrix, inversion of the response matrix
    '''
    return np.linalg.inv(R.T * R + regu * np.matrix(np.eye(R.shape[1]))) * R.T












