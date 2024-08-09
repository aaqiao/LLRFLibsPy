"""Identify the RF system transfer function and characteristic parameters."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for RF system identification
 
Implemented:
    - prbs               : produce the PRBS signal for system identification
    - etfe               : Empirical Transfer Function Estimation (ETFE)
    - half_bw_decay      : calculate the cavity half-bandwidth from the RF pulse decay stage
    - detuning_decay     : calculate the cavity detuning from the RF pulse decay stage
    - cav_drv_est        : estimate the required cavity drive and reflected signals from probe signal
    - cav_par_pulse      : calculate the cavity parameters (half-bandwidth and detuning) within the pulse
                           by directly solving the cavity equation
    - cav_par_pulse_obs  : calculate the cavity parameters (half-bandwidth and detuning) within the pulse
                           using the ADRC observer
    - cav_beam_pulse_obs : calculate the beam drive voltage within the pulse using the ADRC observer
    - cav_observer       : construct the ADRC observer and estimate the denoised cavity voltage and the
                           general disturbances
    - iden_impulse       : identify the impulse response of a real/complex SISO system from data
    - beta_powers        : identify the cavity input coupling factor using steady-state forw/refl powers

To be implemented:
    - Identify the cavity input coupling factor
    - Neural network based system identification for RF system: klystron gain curve,
      klystron output pulse, dynamical modeling
    - Frequeny scanning-based sys ID (generate drive signal of superposition of many frequencies - 
      possibly random frequencies and random phases)
    - identify transfer matrix (least-square, cvx)
    - produce SSB signal waveform (with offset frequency, and with correct number of points for 
      continuous phase if repeated)

Some algorithms are referred to the following books:
S. Simrock and Z. Geng, Low-Level Radio Frequency Systems, Springer, 2022
https://link.springer.com/book/10.1007/978-3-030-94419-3 ("LLRF Book")
#########################################################################
'''
import numpy as np
from scipy import signal

from rf_misc import *

def prbs(n, lower_b = -1.0, upper_b = 1.0):
    '''
    Generate prbs signal (PRBS monic polynomials)
            * ``PRBS3   = x^3 + x^2 + 1``
            * ``PRBS4   = x^4 + x^3 + 1``
            * ``PRBS5   = x^5 + x^3 + 1``
            * ``PRBS6   = x^6 + x^5 + 1``
            * ``PRBS7   = x^7 + x^6 + 1``
            * ``PRBS8   = x^8 + x^6 + x^5 + x^4 + 1``
            * ``PRBS9   = x^9 + x^5 + 1``
            * ``PRBS10  = x^10 + x^7 + 1``
            * ``PRBS11  = x^11 + x^9 + 1``
            * ``PRBS12  = x^12 + x^11 + x^10 + x^4 + 1``
            * ``PRBS13  = x^13 + x^12 + x^11 + x^8 + 1``
            * ``PRBS14  = x^14 + x^13 + x^12 + x^2 + 1``
            * ``PRBS15  = x^15 + x^14 + 1``
            * ``PRBS16  = x^16 + x^14 + x^13 + x^11 + 1``
            * ``PRBS17  = x^17 + x^14 + 1``
            * ``PRBS18  = x^18 + x^11 + 1``
            * ``PRBS19  = x^19 + x^18 + x^17 + x^14 + 1``
            * ``PRBS20  = x^20 + x^3 + 1``
            * ``PRBS23  = x^23 + x^18 + 1``
            * ``PRBS31  = x^31 + x^28 + 1``

    Parameters:
        n:       int, number of point 
        lower_b: float, lower boundary
        upper_b: float, upper boundary
        
    Returns:
        status:  good or not
        data:    numpy array, 1-D array of PRBS signal

    Note:
        The prototype is found here
        https://gist.github.com/Btremaine/d2f861947fe3f49053116e7a679252e9.        
    '''
    # check the input
    if n <= 0 or lower_b >= upper_b:
        return False, None

    # get the number of bit for the PRBS
    q    = np.log2(n + 1)
    nbit = np.floor(q)
    if 2**nbit < q: nbit += 1
    nbit = int(nbit)

    # define the bit ids (start from 0) for XOR
    opbits = {  2:  [1, 0],
                3:  [2, 1], 
                4:  [3, 2], 
                5:  [4, 2],
                6:  [5, 4], 
                7:  [6, 5], 
                8:  [7, 5, 4, 3], 
                9:  [8, 4],
                10: [9, 6], 
                11: [10, 8], 
                12: [11, 10, 9, 3], 
                13: [12, 11, 10, 7],
                14: [13, 12, 11, 1], 
                15: [14, 13], 
                16: [15, 13, 12, 10], 
                17: [16, 13],
                18: [17, 10], 
                19: [18, 17, 16, 13], 
                20: [19, 2],
                21: [20, 18],
                22: [21, 20],
                23: [22, 17], 
                24: [23, 22, 21, 16],
                25: [24, 21],
                26: [25, 5, 1, 0],
                27: [26, 4, 1, 0],
                28: [27, 24],
                29: [28, 26],
                30: [29, 5, 3, 0],
                31: [30, 27],
                32: [31, 21, 1, 0]}

    if nbit not in opbits.keys():
        print('Error: Only support up to 32 bit PRBS!')
        return False, None

    # generate the PRBS signal
    data = np.zeros(n)
    lfsr = 1                                        # init value
    for i in range(n):
        fb = 0
        for bit_id in opbits[nbit]:
            fb ^= get_bit(lfsr, bit_id)             # XOR the selected bits
        lfsr = ((lfsr << 1) + fb) & (2**nbit - 1)   # shifted the buffer left by one bit and append 'fb' 
                                                    # to the last bit (see the note above)
        data[i] = float(fb)

    # normalize the range
    data[data == 0] = lower_b
    data[data == 1] = upper_b 

    return True, data

def etfe(u, y, r = 1, exclude_transient = True, transient_batch_num = 1, fs = 1.0):
    '''
    Implement the Empirical Transfer Function Estimation (ETFE).

    Parameters:
        u:                   numpy array, system input waveform
        y:                   numpy array, system output waveform
        r:                   int, periods (number of batches) in the waveform
        exclude_transient:   boolean, True to exclude the transient (the first 
                               ``transient_batch_num`` batches will be excluded from 
                               calculation)
        transient_batch_num: int, number of transient batches
        fs:                  float, sampling frequency, Hz
        
    Returns:
        status:              boolean, True for success
        freq:                numpy array, frequency vector
        G:                   numpy array (complex), frequency response of the system
        u_fft:               numpy array (complex), spectrum of the input waveform
        y_fft:               numpy array (complex), spectrum of the output waveform
    '''
    # check the input
    if (not u.shape == y.shape) or (r <= 0) or \
       (transient_batch_num < 0) or (fs <= 0.0):
        return (False,) + (None,)*4
    
    if exclude_transient:
        if transient_batch_num >= r:
            transient_batch_num = r
    
    # restructure the data
    N = u.shape[0]                  # total number of data points
    K = int(N / r)                  # number of points of each batch 
    
    u = u.reshape((r, K))           # split to r parts and each part with K samples
    y = y.reshape((r, K))
    
    # remove the transient if there are more than 1 batches in data
    if r > 1:
        if exclude_transient:
            u = u[transient_batch_num:]
            y = y[transient_batch_num:]

    # calculate the transfer function and average
    G = np.zeros(K, dtype = complex)
    n = u.shape[0]
    
    for i in range(n):
        u_fft = np.fft.fft(u[i])
        y_fft = np.fft.fft(y[i])
        G     = G + y_fft / u_fft
        
    G /= n

    # tailor the results to the Nyquist band
    freq   = np.arange(K) / K * fs    
    ids_pf = freq <= 0.5 * fs       # positive frequency IDs
    ids_nf = freq >  0.5 * fs       # negative frequency IDs    
    freq[ids_nf] -= fs              # convert to negative frequency    
    
    freq   = np.append(freq[ids_nf],  freq[ids_pf])
    G      = np.append(G[ids_nf],     G[ids_pf])
    u_fft  = np.append(u_fft[ids_nf], u_fft[ids_pf])
    y_fft  = np.append(y_fft[ids_nf], y_fft[ids_pf])
    
    # return the results
    return True, freq, G, u_fft, y_fft

def half_bw_decay(amp_wf, decay_ids, decay_ide, Ts):
    '''
    Calculate the half-bandwidth of a standing-wave cavity from RF pulse decay.
    
    Refer to LLRF Book section 9.4.3.

    Parameters:
        amp_wf:    numpy array, amplitude waveform
        decay_ids: int, starting index of calc window
        decay_ide: int, ending index of calc window
        Ts:        float, sampling time, s
        
    Returns:
        status:    boolean, 'True' for successful calculation
        half_bw:   float, half bandwidth, rad/s
    '''
    # check the input
    WF = np.array(amp_wf)
    if (WF.shape[0] < 3) or (Ts <= 0.0) or \
       (decay_ids <= 0) or (decay_ide <= decay_ids):
        return False, None

    # do the calculation by fitting
    log_wf  = np.log(WF[decay_ids : decay_ide])       # log scaled amplitude
    t       = np.arange(decay_ide - decay_ids) * Ts   # time vector
    p       = np.polyfit(t, log_wf, 1)                # linear fitting
    half_bw = -p[0]
    
    return True, half_bw
    
def detuning_decay(pha_wf_deg, decay_ids, decay_ide, Ts):
    '''
    Calculate the detuning of a standing-wave cavity from RF pulse decay.
    
    Refer to LLRF Book section 9.4.3.

    Parameters:
        pha_wf_deg: numpy array, phase waveform, deg
        decay_ids:  int, starting index of calc window
        decay_ide:  int, ending index of calc window
        Ts:         float, sampling time, s
        
    Returns:
        status:     boolean, 'True' for successful calculation
        detuning:   float, detuning, rad/s
    '''
    # check the input
    WF = np.array(pha_wf_deg)
    if (WF.shape[0] < 3) or (Ts <= 0.0) or \
       (decay_ids <= 0) or (decay_ide <= decay_ids):
        return False, None

    # do the calculation by fitting
    wf_rad   = np.unwrap(WF[decay_ids : decay_ide] * np.pi / 180.0)     # convert to radian
    t        = np.arange(decay_ide - decay_ids) * Ts                    # time vector
    p        = np.polyfit(t, wf_rad, 1)                                 # linear fitting
    detuning = p[0]
    
    return True, detuning

def cav_drv_est(vc, half_bw, Ts, detuning = 0.0, beta = 1e4):
    '''
    Calculate the theoritical drive waveform for the cavity probe waveform.

    Parameters:
        vc:       numpy array (complex), cavity probe waveform
        half_bw:  float, half bandwidth of the cavity, rad/s
        Ts:       float, sampling time, s
        detuning: float, detuning of the cavity, rad/s
        beta:     float, input coupling factor (needed for NC cavities; 
                   for SC cavities, can use the default value, or you can 
                   specify it if more accurate calibration is needed)
                   
    Returns:
        status:   boolean, success (True) or fail (False)
        vf:       numpy array (complex), estimated cavity drive waveform
        vr:       numpy array (complex), estimated cavity reflected waveform

    Note: 
        1. If the cavity probe signal contains pass-band modes, better
           notch filter them.
        2. The estimate is useful to determine how to move the measured
           forward and reflected signals to align with the timing of the
           cavity probe signal.
    '''
    # check the input
    if (vc.shape[0] < 3) or (Ts <= 0.0) or (half_bw <= 0.0):
        return False, None

    # get the discrete equation of the cavity
    AGc = np.matrix([[-(half_bw - 1j * detuning)]], dtype = complex)        # continous cavity equation
    BGc = np.matrix([[2 * beta * half_bw / (beta + 1)]], dtype = complex)
    CGc = DGc = np.matrix(np.zeros(AGc.shape), dtype = complex)
    AGd, BGd, _, _, _ = signal.cont2discrete((AGc, BGc, CGc, DGc), Ts)      # discretize it
        
    A = AGd[0, 0]           # for SISO system (cavity), only keep the scalar values
    B = BGd[0, 0]

    # calculate the cavity input signal by inversing the discrete equation
    vf = (vc[1:] - A * vc[:-1]) / B
    vf = np.append(vf, vf[0])
    vf = np.roll(vf, 1)
    vr = vc - vf

    '''
    # second implementation - direct discretize the cavity equation
    der_vc = (vc[1:] - vc[:-1]) / Ts
    der_vc = np.append(der_vc, der_vc[0])
    der_vc = np.roll(der_vc, 1)

    vf = (der_vc + (half_bw - 1j*detuning) * vc) * (beta + 1) / (2 * half_bw * beta)
    vr = vc - vf
    '''

    return True, vf, vr

def cav_par_pulse(vc, vf, half_bw, Ts, beta = 1e4):
    '''
    Calculate the half-bandwidth and detuning of a standing-wave cavity 
    within an RF pulse with beam off (directly solve the cavity equation).
    
    Refer to LLRF Book section 9.4.3.

    Parameters:
        vc:      numpy array (complex), cavity probe waveform (reference plane)
        vf:      numpy array (complex), cavity forward waveform (calibrated to the
                  same reference plan as the cavity probe signal)
        half_bw: float, half bandwidth of the cavity (derived from early part 
                  of decay), rad/s
        Ts:      float, sampling time, s
        beta:    float, input coupling factor (needed for NC cavities; 
                  for SC cavities, can use the default value, or you can 
                  specify it if more accurate calibration is needed)
                  
    Returns:
        status:  boolean, success (True) or fail (False)
        wh_pul:  numpy array, half-bandwidth in the pulse, rad/s
        dw_pul:  numpy array, detuning in the pulse, rad/s
    '''
    # check the input
    if (not vc.shape == vf.shape) or (half_bw <= 0.0) or \
       (Ts <= 0.0) or (beta <= 0.0):
        return False, None, None

    # use cavity polar equation
    vd = 2 * beta * vf / (beta + 1)
    vc_amp = np.abs(vc)
    vc_pha = np.angle(vc)
    vd_amp = np.abs(vd)
    vd_pha = np.angle(vd)

    # derivative 
    der_vc_amp = np.gradient(vc_amp, Ts)
    der_vc_pha = np.gradient(vc_pha, Ts)

    # calculate the half-bandwidth and detuning
    wh_pul = (half_bw * vd_amp * np.cos(vc_pha - vd_pha) - der_vc_amp) / vc_amp
    dw_pul = der_vc_pha + half_bw * vd_amp / vc_amp * np.sin(vc_pha - vd_pha)

    return True, wh_pul, dw_pul

def cav_par_pulse_obs(vc, vf, half_bw, Ts, vb = None, beta = 1e4, pole_scale = 50):
    '''
    Calculate the half-bandwidth and detuning of a standing-wave cavity within an RF pulse with 
    beam on/off (use ADRC observer).
    
    Refer to the paper "Geng Z (2017a) Superconducting cavity 
    control and model identification based on active disturbance rejection control. IEEE 
    Trans Nucl Sci 64(3):951-958".

    Parameters:
        vc:          numpy array (complex), cavity probe waveform (reference plane)
        vf:          numpy array (complex), cavity forward waveform (calibrated to the
                      same reference plan as the cavity probe signal)
        half_bw:     float, half bandwidth of the cavity (derived from early part of decay), rad/s
        Ts:          float, sampling time, s
        vb:          numpy array (complex), beam drive waveform (calibrated to the
                      same reference plan as the cavity probe signal)
        beta:        float, input coupling factor (needed for NC cavities; 
                      for SC cavities, can use the default value, or you can 
                      specify it if more accurate calibration is needed)
        pole_scale:  float, scale of the cavity half-bandwidth for the observer pole,
                      it should be tens of times of the closed-loop bandwidth of the cavity
                      
    Returns:
        status:      boolean, success (True) or fail (False)
        wh_pul:      numpy array, half-bandwidth in the pulse, rad/s
        dw_pul:      numpy array, detuning in the pulse, rad/s
        vc_est:      numpy array (complex), cavity probe waveform (estimated by observer)
        f_est:       numpy array (complex), general disturbance WF (estimated by observer)
    '''
    # check the input
    if (not vc.shape == vf.shape) or (half_bw <= 0) or \
       (beta <= 0) or (pole_scale <= 0) or (Ts <= 0):
        return (False,) + (None,)*4

    if vb is not None:
        if not vc.shape == vb.shape:
            return (False,) + (None,)*4

    # get the observer output
    status, vc_est, f_est = cav_observer(vc, vf, half_bw, Ts,
                                         beta       = beta, 
                                         pole_scale = pole_scale)
    if not status:
        return (False,) + (None,)*4

    # calculate the intra-pulse half bandwidth and detuning
    try:
        if vb is None:
            cgain = f_est / vc_est                          # without beam
        else:
            cgain = (f_est - 2 * half_bw * vb) / vc_est     # with beam
    except:
        pass

    cgain  = np.nan_to_num(cgain, copy = False)             # put NaN to 0 (default number)
    wh_pul = -np.real(cgain)
    dw_pul =  np.imag(cgain)

    # return the results
    return True, wh_pul, dw_pul, vc_est, f_est

def cav_beam_pulse_obs(vc, vf, wh_pul, dw_pul, half_bw, Ts, beta = 1e4, pole_scale = 20):
    '''
    Calculate the beam drive waveform of a standing-wave cavity given the intra-pulse half-bandwidth 
    and detuning.
    
    Refer to the paper "Geng Z (2017a) Superconducting cavity control and model identification 
    based on active disturbance rejection control. IEEE Trans Nucl Sci 64(3):951-958".
    
    Parameters:
        vc:          numpy array (complex), cavity probe waveform (reference plane)
        vf:          numpy array (complex), cavity forward waveform (calibrated to
                      the same reference plane as the cavity probe signal)
        wh_pul:      numpy array, half-bandwidth in the pulse, rad/s
        dw_pul:      numpy array, detuning in the pulse, rad/s
        half_bw:     float, half bandwidth of the cavity (derived from early part of decay), rad/s
        Ts:          float, sampling time, s
        beta:        float, input coupling factor (needed for NC cavities; 
                      for SC cavities, can use the default value, or you can 
                      specify it if more accurate calibration is needed)
        pole_scale:  float, scale of the cavity half-bandwidth for the observer pole,
                      it should be tens of times of the closed-loop bandwidth of the cavity
                      
    Returns:
        status:      boolean, success (True) or fail (False)
        vb_est:      numpy array (complex), beam drive waveform (estimated by observer)
        vc_est:      numpy array (complex), cavity probe waveform (estimated by observer)
        f_est:       numpy array (complex), general disturbance WF (estimated by observer)
    '''
    # check the input
    if (not vc.shape == vf.shape == wh_pul.shape == dw_pul.shape) or (half_bw <= 0) or \
       (beta <= 0) or (pole_scale <= 0) or (Ts <= 0):
        return False, None, None, None

    # get the observer output
    status, vc_est, f_est = cav_observer(vc, vf, half_bw, Ts, 
                                         beta       = beta, 
                                         pole_scale = pole_scale)
    if not status:
        return False, None, None, None

    # calculate the beam drive
    b1     = 2 * half_bw
    cgain  = wh_pul + 1j*dw_pul
    vb_est = (f_est + cgain * vc_est) / b1

    # return the results
    return True, vb_est, vc_est, f_est

def cav_observer(vc, vf, half_bw, Ts, beta = 1e4, pole_scale = 50):
    '''
    Estimate the cavity voltage (denoised) and the general disturbance with the ADRC 
    observer.
    
    Refer to the paper "Geng Z (2017a) Superconducting cavity control and model 
    identification based on active disturbance rejection control. IEEE Trans Nucl Sci 64(3):951-958".
        
    Parameters:
        vc:          numpy array (complex), cavity probe waveform (reference plane)
        vf:          numpy array (complex), cavity forward waveform (calibrated to
                      the same reference plane as the cavity probe signal)
        half_bw:     float, half bandwidth of the cavity (derived from early part of decay), rad/s
        Ts:          float, sampling time, s
        beta:        float, input coupling factor (needed for NC cavities; 
                      for SC cavities, can use the default value, or you can 
                      specify it if more accurate calibration is needed)
        pole_scale:  float, scale of the cavity half-bandwidth for the observer pole,
                      it should be tens of times of the closed-loop bandwidth of the cavity
                      
    Returns:
        status:      boolean, success (True) or fail (False)
        vc_est:      numpy array (complex), cavity probe waveform (estimated by observer)
        f_est:       numpy array (complex), general disturbance WF (estimated by observer)
    '''
    # check the input
    if (not vc.shape == vf.shape) or (half_bw <= 0) or \
       (beta <= 0) or (pole_scale <= 0) or (Ts <= 0):
        return False, None, None

    # parameters
    p_obs = -pole_scale * half_bw               # pole of the observer
    m1    = -2 * p_obs                          # observer matrix parameter
    m2    = p_obs**2                            # observer matrix parameter
    b0    = 2 * beta * half_bw / (beta + 1)     # gain for RF drive voltage

    # construct the ADRC observer
    A_obs = np.matrix([[-m1,   0,  1,  0],
                       [  0, -m1,  0,  1],
                       [-m2,   0,  0,  0],
                       [  0, -m2,  0,  0]])
    B_obs = np.matrix([[ m1,   0, b0,  0],
                       [  0,  m1,  0, b0],
                       [ m2,   0,  0,  0],
                       [  0,  m2,  0,  0]])
    C_obs = D_obs = np.matrix(np.zeros(A_obs.shape))

    # simulate the observer output - denoised cavity voltage and general disturbance
    U = np.vstack((np.real(vc), np.imag(vc), np.real(vf), np.imag(vf)))
    T = np.arange(vc.shape[0]) * Ts
    X = signal.lsim((A_obs, B_obs, C_obs, D_obs), U.T, T)
    Y = X[2].T

    # construct the complex signals
    vc_est = Y[0] + 1j*Y[1]
    f_est  = Y[2] + 1j*Y[3]

    return True, vc_est, f_est

def iden_impulse(U, Y, order = 20):
    '''
    Identify the impulse response using the input-output data.

    Parameters:
        U:      numpy array (complex), input waveforms
        Y:      numpy array (complex), output waveforms
        order:  int, order of the impulse response
        
    Returns:
        status: boolean, success (True) or fail (False)
        h:      numpy array (complex), impulse response
    '''
    # check the input
    if (not U.shape == Y.shape) or (order < 2):
        return False, None

    # get the dimension
    n_wf = U.shape[0]       # number of waveforms
    N    = U.shape[1]       # number of points in each waveform

    if order > N:
        order = N - 1       # cannot exceed the point number in a WF
    
    # construct the linear equation
    A = np.zeros(((N - order) * n_wf, order), dtype = complex)
    B = np.zeros( (N - order) * n_wf,         dtype = complex)
    
    for k in range(n_wf):
        for i in range(N - order):
            A[k * (N - order) + i, :] = U[k][i : i + order]
            B[k * (N - order) + i]    = Y[k][i + order]

    # least-square fitting
    X = np.linalg.lstsq(A, B, rcond = None)

    # return the results
    return True, X[0][::-1]

def beta_powers(pf, pr, weak = False):
    '''
    Identify the cavity input coupling factor using the steady-state forward/reflected powers.
    
    Refer to LLRF Book section 9.4.1.

    Parameters:
        pf:     float, forward power in steady-state
        pr:     float, reflected power in steady-state (with the same unit as pf)
        weak:   boolean, True for weak coupling (beta < 1), False for strong coupling (beta > 1)
        
    Returns:
        status: boolean, True for success
        beta:   float, cavity input coupling factor
    '''
    # make physical
    if pr >= pf:
        pr = pf - 1.0e-8

    # check the input
    if pf <= 0 or pr < 0:
        return False, None

    # make the calculation
    alpha = np.sqrt(pr / pf)
    if weak:
        beta = (1.0 - alpha) / (1.0 + alpha)
    else:
        beta = (1.0 + alpha) / (1.0 - alpha)

    return True, beta






















