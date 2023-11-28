"""Measure RF amplitude and phase from ADC samples."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for RF signal detection and actuation

Assumptions:
  a. If not explicitly mentioned, all phases are in degree
 
Implemented:
    - noniq_demod   : perform non-I/Q demodulation of a given raw sampling waveform
    - twop_demod    : demodulate raw with every two samples
    - asyn_demod    : demodulate raw sampled by asyn. clock, reference WF needed
    - self_demod_ap : demodulate raw with Hilbert transform, return amplitude and phase
    - iq2ap_wf      : convert I/Q waveforms to amplitude/phase waveforms
    - ap2iq_wf      : convert amplitude/phase waveforms to I/Q waveforms
    - norm_phase    : normalize phase (scalar or WF) to a specific range (default +-180 deg)
    - pulse_info    : derive the pulse info like pulse width, pulse offset, etc.

Some algorithms are referred to the following books:
S. Simrock and Z. Geng, Low-Level Radio Frequency Systems, Springer, 2022
https://link.springer.com/book/10.1007/978-3-030-94419-3 ("LLRF Book")
#########################################################################
'''
import numpy as np
from scipy import signal

def noniq_demod(raw_wf, n, m = 1):
    '''
    Non-I/Q demodulation.

    Refer to LLRF Book section 5.2.2.
    
    Parameters:
        raw_wf: numpy array, 1-D array of the raw waveform
        m, n:   integer, non-I/Q parameters (n samples cover m IF cycles)
        
    Returns:
        status: boolean, success (True) or faile (False)
        I, Q:   numpy array, I/Q waveforms
    '''
    # check the input
    L = raw_wf.shape[0]
    if n <= 0 or m <= 0 or n <= m or L < n:
        return False, None, None

    # calculate the demod coefficients
    P_rad   = np.arange(0.0, n, 1) * 2.0 * m * np.pi / n    # phases of one superperiod of NCO
    I_coef  = np.sin(P_rad) * 2.0 / n                       # NCO output I
    Q_coef  = np.cos(P_rad) * 2.0 / n                       # NCO output Q

    # define the databuffers
    I       = np.zeros(L)       # final output buffer
    Q       = np.zeros(L)
    pipeI   = np.zeros(n + 1)   # FIFO helping calculation
    pipeQ   = np.zeros(n + 1)
    sumI    = 0.0               # accumulators
    sumQ    = 0.0
    coef_id = 0                 # index for data selection
    pipe_id = n

    # do demodulation
    for i in range(L):
        # fill the FIFO a new multiplication
        pipeI[pipe_id] = raw_wf[i] * I_coef[coef_id]
        pipeQ[pipe_id] = raw_wf[i] * Q_coef[coef_id]

        # calculate difference btw FIFO new/old, and accumulate
        sumI += pipeI[pipe_id] - pipeI[pipe_id-1]
        sumQ += pipeQ[pipe_id] - pipeQ[pipe_id-1]

        # update the output for an sample
        I[i] = sumI
        Q[i] = sumQ

        # update the index
        coef_id += 1
        if coef_id >= n:
            coef_id = 0

        pipe_id -= 1
        if pipe_id < 0:
            pipe_id = n

    # return the results
    return True, I, Q

def twop_demod(raw_wf, f_if, fs):
    '''
    Demodulation with two points.
    
    Refer to LLRF Book section 5.2.2.

    Parameters:
        raw_wf: numpy array, raw waveform to be demodulated
        f_if:   float, IF frequency, Hz
        fs:     float, sampling frequency, Hz
        
    Returns:
        status: boolean, success (True) or faile (False)
        I, Q:   numpy array, I/Q waveforms
    '''
    # check the input
    if (raw_wf.shape[0] < 3) or (f_if <= 0.0) or (fs <= 0.0):
        return False, None, None

    # reserve the buffer
    N = raw_wf.shape[0]                 # number of the points in WF
    I = np.zeros(N)
    Q = np.zeros(N)

    # make the demodulation    
    dphi_rad = 2.0 * np.pi * f_if / fs  # phase advance per sample
    sn_dphi  = np.sin(dphi_rad)    

    for i in range(1, N):
        I[i] = ( raw_wf[i] * np.cos((i-1)*dphi_rad) - raw_wf[i-1] * np.cos(i*dphi_rad)) / sn_dphi
        Q[i] = (-raw_wf[i] * np.sin((i-1)*dphi_rad) + raw_wf[i-1] * np.sin(i*dphi_rad)) / sn_dphi

    return True, I, Q

def asyn_demod(raw_wf, ref_wf):
    '''
    Asynchronous demodulation (need reference).

    Parameters:
        raw_wf:      numpy array, signal waveform to be demodulated
        ref_wf:      numpy array, samples of the RF reference signal
    Returns:
        status:      boolean, success (True) or faile (False)
        I, Q:        numpy array, I/Q waveforms of final demodulation
        Isig, Qsig:  numpy array, I/Q waveforms of raw_wf (with inaccurate phase)
        Iref, Qsig:  numpy array, I/Q waveforms of ref_wf (with inaccurate phase)
    '''
    # check the input
    if (not raw_wf.shape == ref_wf.shape) or (raw_wf.shape[0] < 3):
        return (False,) + (None,)*6

    # estimate the reference frequency 
    N = raw_wf.shape[0]                         # number of the points in WF
    Y = np.fft.fft(ref_wf)
    f_if = np.argmax(np.abs(Y[:round(N/2)]))    # find the peak, respresenting the IF freq
    fs   = N                                    # represent the sampling frequency

    # demodulate with twop_demod
    status1, Isig, Qsig = twop_demod(raw_wf, f_if, fs)
    status2, Iref, Qref = twop_demod(ref_wf, f_if, fs)
    if not (status1 and status2):
        return (False,) + (None,)*6

    # get the reference phase
    status, Aref, Pref_deg = iq2ap_wf(Iref, Qref)
    if not status:
        return (False,) + (None,)*6

    # unwrap the reference phase
    Pref_deg = np.unwrap(Pref_deg, period = 360.0)

    # get the linear fitting of the phase (exclude first points)
    T = np.arange(N)   
    p = np.polyfit(T[2:], Pref_deg[2:], 1)

    # reconstruct the clean phase of reference
    Pref_rec_deg  = np.polyval(p, T)
    Pref_rec_deg -= Pref_rec_deg[0]

    # remove the phase slope from the signal phase
    C = (Isig + 1j*Qsig) * np.exp(-1j * Pref_rec_deg * np.pi / 180.0)

    return True, np.real(C), np.imag(C), Isig, Qsig, Iref, Qref

def self_demod_ap(raw_wf, n = 1):
    '''
    Self demodulation with Hilbert transform (later may implement padding to
    mitigate the edge effects).

    Parameters:
        raw_wf: numpy array, signal waveform to be demodulated
        n:      int, number of points covering full cycles (for coherent sampling)
    Returns:
        status: boolean, success (True) or faile (False)
        A, P:   numpy array, amplitude and phase waveforms, P in degree
    '''
    # check the input
    if raw_wf.shape[0] < 3:
        return False, None, None

    # perform Hilbert transform
    N = raw_wf.shape[0]             # number of points in WF
    L = int(N / n) * n              # tailor
    C = signal.hilbert(raw_wf[:L])  # hilbert transform
    
    # to the correct length
    if N > L:
        C = np.hstack((C, np.zeros(N - L)))

    # get the amplitude and phase in deg
    A = np.abs(C)
    P = np.angle(C, deg = True)

    return True, A, P

def iq2ap_wf(I, Q):
    '''
    I/Q to A/P waveforms.
    
    Parameters:
        I, Q: numpy array, I/Q waveforms
        
    Returns:
        A, P: numpy array, amplitude/phase waveforms, P in degree
    '''
    # check the input
    if not I.shape == Q.shape:
        return False, None, None

    # convert to complex
    C = I + 1j*Q

    # return the results
    return True, np.abs(C), np.angle(C, deg = True)

def ap2iq_wf(A, P):
    '''
    A/P to I/Q waveforms.
    
    Parameters:
        A, P: numpy array, amplitude/phase waveforms, P in degree
        
    Returns:
        I, Q: numpy array, I/Q waveforms
    '''
    # check the input
    if not A.shape == P.shape:
        return False, None, None

    # do calculation
    P_rad = P * np.pi / 180.0
    I = A * np.cos(P_rad)
    Q = A * np.sin(P_rad)

    # return the results
    return True, I, Q

def norm_phase(P, cmd = '+-180'):
    '''
    normalize phase to +-180 degree or between 0 to 360 degree.
    
    Parameters:
        P:     float, phase in degree
        cmd:   string '+-180' or '0to360'
    Returns:
        Pnorm: float, normalized phase, degree
    '''
    # convert using the complex numbers
    Pnorm = np.angle(np.exp(1j * P * np.pi / 180.0), deg = True)    # to +-180 deg
    if cmd == '0to360':
        if isinstance(Pnorm, float):
            Pnorm = (Pnorm + 360.0) if (Pnorm < 0.0) else Pnorm
        else:
            Pnorm[Pnorm < 0.0] += 360

    # return the result
    return Pnorm

def pulse_info(pulse, threshold = 0.1):
    '''
    get information of a pulse.

    Parameters:
        pulse:     numpy array, the pulse data
        threshold: float, threshold for edge detection 
    Returns:
        offs:      int, offset id of the pulse
        pulw:      int, pulse width as number of samples
        peak:      float, peak magnitude of the pulse

    To be done:
        a. detect the rise time and falling time
        b. detect the flattop region and average
        c. calculate the energy in the pulse
    '''
    result = {'status': False}

    # check the input
    if isinstance(pulse, list):
        data = np.array(pulse)
    elif isinstance(pulse, np.ndarray):
        data = pulse
    else:
        return result

    if data.shape[0] <= 1 or threshold < 0.0:
        return result

    # be sure the pulse is positive
    data = np.abs(data)

    # get the peak
    result['peak'] = np.max(data)           # peak value of the pulse
    low_l = threshold * result['peak']      # the low limit of the pulse

    # get the pulse info
    over_lowl = data > low_l
    n         = over_lowl.shape[0]          # size of the pulse
    start_id  = 0                           # init the pulse start id
    end_id    = n - 1                       # init the pulse end id

    for i in range(2, n - 1):
        # find the pulse start id
        if (not any(over_lowl[:i])) and all(over_lowl[i:i+2]):
            start_id = i
    
        # find the pulse end id
        if (not any(over_lowl[i:])) and all(over_lowl[i-2:i]):
            end_id = i - 1

    result['offs']   = start_id
    result['pulw']   = end_id - start_id + 1
    result['status'] = True

    # return the results
    return result





























