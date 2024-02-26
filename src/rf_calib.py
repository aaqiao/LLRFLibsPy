"""RF calibrations like virtual probe, RF actuator offset/imbalance, forward and reflected, and power calibrations."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for RF system calibration

Implemented:    
    - calib_vprobe       : calib. cavity virtual probe with forward/reflected signals
    - calib_dac_offs     : calib. DAC offset with I/Q modulator direct upconversion
    - calib_iqmod        : calib. I/Q modulator imbalance with direct upconversion
    - calib_cav_for      : calib. cavity forward signal using forward and probe
    - calib_ncav_for_ref : calib. cavity forward/reflected signals with constant QL and detuning
    - calib_scav_for_ref : calib. cavity forward/reflected signals with time-varying QL and detuning
    - for_ref_volt2power : calib. forward/reflected power from forward/reflected voltage
    - phasing_energy     : calib. energy gain and beam phase with phase scan and energy meas.
    - egain_cav_power    : est. steady-state standing-wave cavity voltage from drive power
    - egain_cgstr_power  : est. const gradient traveling-wave structure ACC voltage from drive power
    - calib_vsum_poor    : calib. vector sum by rotating and scaling referring to one channel
    - calib_sys_gain_pha : calib. system gain and system phase

To be implemented: 
    - calibrate the Vacc and phi_b of each cavity using beam transient
    - calibrate the vector sum (beam transient based)

Some algorithms are referred to the following books:
S. Simrock and Z. Geng, Low-Level Radio Frequency Systems, Springer, 2022
https://link.springer.com/book/10.1007/978-3-030-94419-3 ("LLRF Book")
#########################################################################
'''
import numpy as np

from rf_sysid import *
from rf_fit import *

def calib_vprobe(vc, vf_m, vr_m):
    '''
    Calibrate the virtual probe signal of the cavity with ``vc = m * vf_m + n * vr_m``
    aiming at reconstructing the probe signal with the forward and reflected
    signals. This is useful when the probe is temperally not available.
    
    Parameters:
         vc:   numpy array (complex), cavity probe waveform (as reference plane)
         vf_m: numpy array (complex), cavity forward waveform (meas.)
         vr_m: numpy array (complex), cavity reflected waveform (meas.)
         
    Returns:
        status: boolean, success (True) or fail (False)
        m, n:   float (complex), calibration coefficients    
    
    Note: 
        The waveforms should have been aligned in time, i.e., the relative delays between them should have been removed.
    '''
    # check the input
    if (not (vc.shape == vf_m.shape == vr_m.shape)) or (len(vc) < 3):
        return False, None, None

    # formulate the problem of Ax = B
    X = np.linalg.lstsq(np.vstack((vf_m, vr_m)).T, vc, rcond = None)

    # return the status, m, n
    return True, X[0][0], X[0][1]

def calib_dac_offs(vdac, viqm, sig_ids, sig_ide, leak_ids, leak_ide, delay = 0):
    '''
    Calibrate the DAC offset to remove carrier leakage for direct upconversion.
    
    Refer to LLRF Book section 9.2.2.
    
    Parameters:
        vdac:     numpy array (complex), DAC output waveform
        viqm:     numpy array (complex), I/Q modulator output meas. waveform
        sig_ids:  int, starting index of signals
        sig_ide:  int, ending index of signals
        leak_ids: int, starting index of leakage
        leak_ide: int, ending index of leakage
        delay :   int, delay in number of points between I/Q out and DAC WFs,
                        it is needed to be set reflecting the actual delay for
                        better results
    Returns:
        status:   boolean, success (True) or fail (False)
        offset_I: float, I offset to be added to the actual DAC output
        offset_Q: float, Q offset to be added to the actual DAC output
    '''
    # check the input
    N = vdac.shape[0]
    if vdac.shape != viqm.shape or \
       sig_ids  < 0 or sig_ide  < 0 or \
       leak_ids < 0 or leak_ide < 0 or \
       (sig_ids  + delay) >= N or (sig_ide  + delay) >= N or \
       (leak_ids + delay) >= N or (leak_ide + delay) >= N or \
       sig_ids >= sig_ide or leak_ids >= leak_ide:
        return False, None, None

    # make the calculation of the offset
    vdac_sig  = np.mean(vdac[sig_ids  : sig_ide])
    vdac_leak = np.mean(vdac[leak_ids : leak_ide])
    viqm_sig  = np.mean(viqm[sig_ids  + delay : sig_ide  + delay])
    viqm_leak = np.mean(viqm[leak_ids + delay : leak_ide + delay])

    voff = -(vdac_sig - vdac_leak) / (viqm_sig - viqm_leak) * viqm_leak
    
    return True, np.real(voff), np.imag(voff)

def calib_iqmod(vdac, viqm):
    '''
    Calibrate the I/Q modulator imbalance, return ``inv(M)`` with 
    ``[real(viqm) imag(viqm)]' = M * [real(vdac) imag(vdac)]' + [iqm_offs_I iqm_offs_Q]'``
    This method is OK to calibrate the imbalance matrix M (use its inverse
    to pre-distort the DAC output) but the offset is not accurate to determine
    the DAC offset calibration. It is suggested to calibrate the DAC offset
    first using the routine ``calib_dac_offs`` before executing this function.
    
    Refer to paper "Geng Z, Hong B (2016) Design and calibration of an RF actuator for 
    low-level RF systems. IEEE Trans Nucl Sci 63(1):281-287".
 
    Parameters:
        vdac: numpy array (complex), DAC actuation points
        viqm: numpy array (complex), I/Q modulator meas. points
        
    Returns:
        status: boolean, success (True) or fail (False)
        invM:   numpy matrix, inversion of M, can be directly used to 
                    correct the I/Q modulator imbalance
        viqm_s: numpy array (complex), scaled I/Q modulator output
        viqm_c: numpy array (complex), re-constructed I/Q mod. output
    '''
    # check the input
    if vdac.shape != viqm.shape or vdac.shape[0] < 3:
        return (False,) + (None,)*3

    # scale & rotate the I/Q modulator output to align with DAC
    viqm_s = np.mean(vdac/viqm) * viqm

    # calibrate with least-square method
    Y = np.matrix(np.vstack((np.real(viqm_s), np.imag(viqm_s))))
    U = np.matrix(np.vstack((np.real(vdac), np.imag(vdac), np.ones(vdac.shape[0]))))
    M = Y * U.T * np.linalg.inv(U * U.T)

    # reconstruct the scaled output
    Y_t = np.array(M * U)
    viqm_c = Y_t[0] + 1j*Y_t[1]

    return True, np.linalg.inv(M[:2, :2]), viqm_s, viqm_c

def calib_cav_for(vc, vf_m, pul_ids, pul_ide, half_bw, detuning, Ts, beta = 1e4):
    '''
    Calibrate the cavity forward signal: ``vf = a * vf_m``. The resulting ``vf`` is
    the cavity drive signal at the same reference plane as the cavity probe ``vc``.
    
    Refer to LLRF Book section 4.6.1.3.
    
    Parameters:
        vc:       numpy array (complex), cavity probe waveform (reference plane)
        vf_m:     numpy array (complex), cavity forward waveform (meas.)
        pul_ids:  int, starting index for calculation (calculation window should 
                   cover the entire pulse for NC cavities; use the part close to 
                   pulse end for SC cavities)
        pul_ide:  int, ending index for calculation 
        half_bw:  float, half bandwidth of the cavity, rad/s
        detuning: float, detuning of the cavity, rad/s
        Ts:       float, sampling time, s
        beta:     float, input coupling factor (needed for NC cavities; 
                   for SC cavities, can use the default value, or you can 
                   specify it if more accurate calibration is needed)
                   
    Returns:
        status:   boolean, success (True) or fail (False)
        a:        float (complex), calibrate coefficient
    
    Note: 
        The waveforms should have been aligned in time, i.e., the relative delays between them should have been removed.
    '''
    # check the input
    if (not (vc.shape == vf_m.shape)) or (len(vc) < 3) or \
       (pul_ids < 0) or (pul_ide <= pul_ids) or (Ts <= 0) or \
       (half_bw <= 0):
        return False, None

    # estimate the theoritical forward from probe
    status, vf_est, _ = cav_drv_est(vc, half_bw, Ts, detuning, beta)
    if not status:
        return False, None

    # make the calibration
    return True, np.mean(vf_est[pul_ids:pul_ide]) / np.mean(vf_m[pul_ids:pul_ide])

def calib_ncav_for_ref(vc, vf_m, vr_m, pul_ids, pul_ide, half_bw, detuning, Ts, beta = 1.0):
    '''
    Calibrate the cavity forward and reflected signals for a normal conducting 
    cavity with constant loaded Q and detuning:
    ``vf = a * vf_m + b * vr_m, vr = c * vf_m + d * vr_m``
    The resulting ``vf`` and ``vr`` are the cavity drive/reflected signals at the 
    same reference plane as the cavity probe ``vc``.
    
    Refer to LLRF Book section 9.3.5.
    
    Parameters:
        vc:        numpy array (complex), cavity probe waveform (reference plane)
        vf_m:      numpy array (complex), cavity forward waveform (meas.)
        vr_m:      numpy array (complex), cavity reflected waveform (meas.)
        pul_ids:   int, starting index for calculation (calculation window should 
                    cover entire pulse)
        pul_ide:   int, ending index for calculation
        half_bw:   float, half bandwidth of the cavity, rad/s
        detuning:  float, detuning of the cavity, rad/s
        Ts:        float, sampling time, s
        beta:      float, input coupling factor (needed for NC cavities)
        
    Returns:
        status:    boolean, success (True) or fail (False)
        a,b,c,d:   float (complex), calibrate coefficients
    
    Note: 
        The waveforms should have been aligned in time, i.e., the relative delays between them should have been removed.
    '''    
    # check the input
    if (not (vc.shape == vf_m.shape == vr_m.shape)) or (len(vc) < 3) or \
       (pul_ids < 0) or (pul_ide <= pul_ids) or (Ts <= 0) or \
       (half_bw <= 0):
        return (False,) + (None,)*4

    # estimate the theoritical forward and reflected signals from probe
    status, vf_est, vr_est = cav_drv_est(vc, half_bw, Ts, detuning, beta)
    if not status:
        return (False,) + (None,)*4

    # calibrate the virtual probe to get: vc = m * vf_m + n * vr_m
    # with m = a + c, n = b + d
    status, m, n = calib_vprobe(vc, vf_m, vr_m)
    if not status:
        return (False,) + (None,)*4

    # use "calib_vprobe" which solves a multi-linear fitting problem
    status, a, b = calib_vprobe(vf_est[pul_ids:pul_ide], 
                                vf_m[pul_ids:pul_ide], 
                                vr_m[pul_ids:pul_ide])
    if not status:
        return (False,) + (None,)*4

    # finally return the results
    return True, a, b, m - a, n - b

def calib_scav_for_ref(vc, vf_m, vr_m, pul_ids, pul_ide, decay_ids, decay_ide, 
                       half_bw, detuning, Ts, beta = 1e4):
    '''
    Calibrate the cavity forward and reflected signals for a superconducting 
    cavity with time-varying loaded Q or detuning:
    ``vf = a * vf_m + b * vr_m, vr = c * vf_m + d * vr_m``
    The resulting ``vf`` and ``vr`` are the cavity drive/reflected signals at the 
    same reference plane as the cavity probe ``vc``.
    
    Refer to LLRF Book section 9.3.5.
    
    Parameters:
        vc:        numpy array (complex), cavity probe waveform (reference plane)
        vf_m:      numpy array (complex), cavity forward waveform (meas.)
        vr_m:      numpy array (complex), cavity reflected waveform (meas.)
        pul_ids:   int, starting index for calculation (calculation window should 
                    take the pulse end part close to decay)
        pul_ide:   int, ending index for calculation
        decay_ids: int, starting id of calculation window at decay stage
        decay_ide: int, ending id of calculation window at decay stage
        half_bw:   float, half bandwidth of the cavity (derived from early part of decay), rad/s
        detuning:  float, detuning of the cavity (derived from early part of decay), rad/s
        Ts:        float, sampling time, s
        beta:      float, input coupling factor (for SC cavities, can use the default value,
                    or you can specify it if more accurate calibration is needed)
                    
    Returns:
        status:    boolean, success (True) or fail (False)
        a,b,c,d:   float (complex), calibrate coefficients
    
    Note: 
        The waveforms should have been aligned in time, i.e., the relative delays between them should have been removed.
    '''
    # check the input
    if (not (vc.shape == vf_m.shape == vr_m.shape)) or (len(vc) < 3) or \
       (pul_ids < 0) or (pul_ide <= pul_ids) or (Ts <= 0) or \
       (decay_ids < 0) or (decay_ide <= decay_ids) or (half_bw <= 0):
        return (False,) + (None,)*4

    # estimate the theoritical forward and reflected signals from probe
    status, vf_est, vr_est = cav_drv_est(vc, half_bw, Ts, detuning, beta)
    if not status:
        return (False,) + (None,)*4

    # calibrate the virtual probe to get: vc = m * vf_m + n * vr_m
    # with m = a + c, n = b + d
    status, m, n = calib_vprobe(vc, vf_m, vr_m)
    if not status:
        return (False,) + (None,)*4        

    # find from decay z = b/a
    z = -np.mean(vf_m[decay_ids:decay_ide]) / np.mean(vr_m[decay_ids:decay_ide])

    # find a good "a" to match the estimated forward signal close to pulse end
    #   vf_est = a * vf_m + a * z * vr_m
    a = np.mean(vf_est[pul_ids:pul_ide]) / np.mean(vf_m[pul_ids:pul_ide] + z * vr_m[pul_ids:pul_ide])
    
    # get other parameters
    b = a * z
    c = m - a
    d = n - b

    # finally return the results
    return True, a, b, c, d

def for_ref_volt2power(roQ_or_RoQ, QL, 
                       vf_pcal  = None, 
                       vr_pcal  = None, 
                       beta     = 1e4, 
                       machine  = 'linac'):
    '''
    Convert the calibrated forward and reflected signals (in physical unit, V)
    to forward and reflected power (in W).
    
    Refer to LLRF Book section 3.3.9.
    
    Parameters:
        roQ_or_RoQ:  float, cavity r/Q of Linac or R/Q of circular accelerator, Ohm
                      (see the note below)
        QL:          float, loaded quality factor of the cavity
        vf_pcal:     numpy array (complex), forward waveform (calibrated to physical unit)
        vf_pcal:     numpy array (complex), reflected waveform (calibrated to physical unit)
        beta:        float, input coupling factor (needed for NC cavities; 
                      for SC cavities, can use the default value, or you can 
                      specify it if more accurate result is needed)
        machine:     string, 'linac' or 'circular', used to select r/Q or R/Q
        
    Returns:
        status:      boolean, success (True) or fail (False)
        for_power:   numpy array, waveform of forward power (if input is not None), W
        ref_power:   numpy array, waveform of reflected power (if input is not None), W
        C:           float (complex), calibration coefficient: power_W = C * Volt_V^2
    
    Note: 
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines 
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to 
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (roQ_or_RoQ <= 0.0) or (QL <= 0.0) or (beta <= 0.0):
        return (False,) + (None,)*3

    # calculate the loaded resistance
    if machine == 'circular':
        RL = roQ_or_RoQ * QL
    else:
        RL = 0.5 * roQ_or_RoQ * QL

    # calculate the coefficient to convert voltage to power
    C = beta / (beta + 1) / (2 * RL)

    # convert the voltage to power if they are valid
    for_power = ref_power = None
    if vf_pcal is not None: for_power = C * np.abs(vf_pcal)**2
    if vr_pcal is not None: ref_power = C * np.abs(vr_pcal)**2

    return True, for_power, ref_power, C

def phasing_energy(phi_vec, E_vec, machine  = 'linac'):
    '''
    Calibrate the beam phase using the beam energy measured by scanning RF phase.
    
    Refer to LLRF Book section 9.3.2.3.
    
    Parameters:
        phi_vec: numpy array, phase measurement, degree
        E_vec:   numpy array, beam energy measurement
        machine: string, 'linac' or 'circular', see the note below
        
    Returns:
        status:  boolean, success (True) or fail (False)
        Egain:   float, maximum energy gain of the RF station
        phi_off: float, phase offset that should be added to the phase meas, deg
    
    Note: 
          Circular accelerator use sine as convention of beam phase definition,
          resulting in 90 degree for on-crest acceleration; while for Linacs, 
          the consine convention is used with 0 degree for on-crest acceleration.
    '''
    # check the input
    if (not phi_vec.shape == E_vec.shape) or (phi_vec.shape[0] < 3):
        return False, None, None

    # make the fitting
    if machine == 'circular':
        status, Egain, phi_rad, c = fit_sincos(phi_vec*np.pi/180.0, E_vec, target = 'sin')
    else:
        status, Egain, phi_rad, c = fit_sincos(phi_vec*np.pi/180.0, E_vec, target = 'cos')

    if not status:
        return False, None, None

    # return the energy gain and phase offset
    return True, Egain, phi_rad*180.0/np.pi

def egain_cav_power(pfor, roQ_or_RoQ, QL, beta = 1e4, machine = 'linac'):
    '''
    Standing-wave cavity energy gain estimate from RF drive power without beam loading.
    
    Refer to LLRF Book section 9.3.2.1, equation (9.13).
    
    Parameters:
        pfor:        float, cavity drive power, W
        roQ_or_RoQ:  float, cavity r/Q of Linac or R/Q of circular accelerator, Ohm
                      (see the note below)
        QL:          float, loaded quality factor of the cavity
        beta:        float, input coupling factor (needed for NC cavities; 
                      for SC cavities, can use the default value, or you can 
                      specify it if more accurate result is needed)
        machine:     string, ``linac`` or ``circular``, used to select r/Q or R/Q 
    Returns:
        status:      boolean, success (True) or fail (False)
        vc0:         float, desired cavity voltage for the given drive power
    
    Note: 
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines 
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to 
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (pfor < 0.0) or (roQ_or_RoQ <= 0.0) or (QL <= 0.0) or (beta <= 0.0):
        return False, None

    # calculate the loaded resistance
    if machine == 'circular':
        RL = roQ_or_RoQ * QL
    else:
        RL = 0.5 * roQ_or_RoQ * QL

    # calculate the maximum energy gain (accelerating voltage)
    vc0 = 2.0 * np.sqrt(2.0 * beta * RL * pfor / (beta + 1.0))   
    return True, vc0

def egain_cgstr_power(pfor, f0, rs, L, Q, Tf):
    '''
    Traveling-wave constant gradient structure energy gain estimate from RF 
    drive power without beam loading.
    
    Refer to LLRF Book section 9.3.2.1, equation (9.14).
    
    Parameters:
        pfor: float, structure input RF power, W
        f0:   float, RF operating frequency, Hz
        rs:   float, shunt impedance per unit length, Ohm/m
        L:    float, length of the traveling-wave structure, m
        Q:    float, quality factor of the TW strcuture
        Tf:   float, filling time of the TW structure, s

    Returns:
        status: boolean, success (True) or fail (False)
        vc0:    float, desired accelerating voltage for the given drive power
    '''
    # check the input
    if (f0 <= 0.0) or (pfor < 0.0) or (rs <= 0.0) or (L <= 0.0) or \
       (Q <= 0.0) or (Tf <= 0.0):
        return False, None

    # calculate the total power attenuation factor
    tao_tw = Tf * np.pi * f0 / Q

    # calculate the maximum energy gain (accelerating voltage)
    vacc0 = np.sqrt(L * pfor * rs * (1.0 - np.exp(-2.0 * tao_tw)))
    return True, vacc0

def calib_vsum_poor(amp_vec, pha_vec, ref_ch = 0):
    '''
    Calibrate the vector sum (poor man's solution by rotating and scaling all other 
    channels refering to the first channel).
    
    Parameters:
        amp_vec: numpy array, amplitude of all channels
        pha_vec: numpy array, phase of all channels, deg
        ref_ch:  int, reference channel, between 0 and (num of channels - 1)
        
    Returns:
        status:  boolean, success (True) or fail (False)
        scale:   numpy array, scale factor for all channels
        phase:   numpy array, phase offset adding to all channels, deg
    '''
    # check the input
    if (not amp_vec.shape == pha_vec.shape) or \
       (ref_ch < 0 or ref_ch >= amp_vec.shape[0]):
        return False, None, None

    # make the calculation
    scale = amp_vec[ref_ch] / amp_vec
    phase = pha_vec[ref_ch] - pha_vec

    return True, scale, phase

def calib_sys_gain_pha(vact, vf, beta = 1e4):
    '''
    Calibrate the system gain and sytem phase.
    
    Refer to LLRF Book section 9.4.2.
    
    Parameters:
        vact:      numpy array (complex), RF actuation waveform (usually DAC output)
        vf:        numpy array (complex), cavity forward signal (calibrated to 
                    the reference plane of the cavity voltage or vector-sum voltage)
        beta:      float, input coupling factor (needed for NC cavities; 
                    for SC cavities, can use the default value, or you can 
                    specify it if more accurate result is needed)    
    Returns:
        status:    boolean, success (True) or fail (False)
        sys_gain:  numpy array, waveform of system gain
        sys_phase: numpy array, waveform of system phase, deg
    '''
    # check the input
    if (not vact.shape == vf.shape) or (beta <= 0.0):
        return False, None, None

    # make the calibration
    G         = 2.0 * beta / (beta + 1.0) * vf / vact
    sys_gain  =  np.abs(G)
    sys_phase = -np.angle(G, deg = True)

    # return the results
    return True, sys_gain, sys_phase



















