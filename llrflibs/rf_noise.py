"""Analyze, generate and filter noise."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for RF noise analysis
 
Implemented:
    - calc_psd_coherent   : calculate the power spectral density (PSD) of a 
                            coherent sampled data series
    - calc_psd            : calculate the PSD of a general data series
    - rand_unif           : generate uniform distributed random numbers
    - gen_noise_from_psd  : generate noise series from a given PSD spectrum
    - calc_rms_from_psd   : calculate the RMS jitter from a given PSD spectrum
    - notch_filt          : apply notch filter to a data series
    - design_notch_filter : design a notch filter in state-space format
    - filt_step           : apply a single time step of state-space filter
    - rand_sine           : generate random sine signals

To be implemented:
    - correlation
    - sum and sub of dB values
    - random sub/sum (RMS values)
    - moving average filter (with group delay compensated)
    - estimate the RF detector nonlinearity caused meas. errors

Some algorithms are referred to the following books:
S. Simrock and Z. Geng, Low-Level Radio Frequency Systems, Springer, 2022
https://link.springer.com/book/10.1007/978-3-030-94419-3 ("LLRF Book")
#########################################################################
'''
import numpy as np
from scipy import signal

def calc_psd_coherent(data, fs, bit = 0, n_noniq = 1, plot = False):
    '''
    Calculate the power spectral density of the input waveform (coherent sampling).
    
    Refer to LLRF Book section 6.1.2.

    Parameters:
        data:        numpy array, 1-D array of the raw waveform
        fs:          float, sampling frequency, Hz
        bit:         int, number of bit of data. If bit = 0, do not scale data
        n_noniq:     int, non-I/Q parameters (n samples cover a full cycle)
        plot:        boolean, True for plot the spectrum
        
    Returns:
        freq:        numpy array, frequency waveform, Hz
        amp_resp:    numpy array, amplitude response, dBFS/Hz
        pha_resp:    numpy array, phase response, degree
        signal_freq: float, signal frequency, Hz
        signal_mag:  float, signal level, dBFS
        noise_flr:   float, noise floor, dBFS/Hz
        snr:         float, signal-to-noise ratio, dB
        bin_db:      float, offset for an FFT bin freq space, dB
        status:      boolean, success (True) or fail (False)

    Note:
        The unit dBFS is for ADC raw data, if for phase in radian, it should be
        replaced by dBrad^2; if for relative amplitude, should be replaced by dB.
    '''
    # results
    result = {'status': False}

    # get the input data size
    if isinstance(data, np.ndarray): L = data.shape[0]
    elif isinstance(data, list):     L = len(data)
    else:                            L = 0

    # tailor the data: multiple of n_noniq (coherent)
    if n_noniq <= 0: n_noniq = 1                        # avoid dividing 0
    N = int(L / n_noniq) * n_noniq

    # check the input
    if N < 3 or fs <= 0 or bit < 0:
        return result

    # calculate the FFT and get the results in Nyquist band
    if bit > 0:
        fftd = data[0:N] / (2.0**(bit-1)/np.sqrt(2.0))  # tailor the data and normalize with full scale
    else:
        fftd = data[0:N]

    Y    = np.fft.fft(fftd)                             # do FFT
    freq = np.arange(N) / N * fs                        # frequency vector, Hz
    ids  = freq <= 0.5 * fs                             # index of Nyquist band
    Y    = Y[ids]                                       # tailor the response
    freq = freq[ids]                                    # tailor the frequency vector

    # calculate the PSD
    PSD  = np.abs(Y) * np.abs(Y) / (N * fs)             # SSB spectra, linear scale relative power / Hz
    if (N % 2) == 0: PSD[1:-1] *= 2.0                   # convert to DSB spectra (exclude DC and f_nyquist)
    else:            PSD[1:]   *= 2.0                   # convert to DSB spectra (exclude DC)

    # calculate the derived quantities
    amp_resp    = 10.0 * np.log10(PSD)                  # in dBFS/Hz
    pha_resp    = np.angle(Y, deg = True)               # in degree
    bin_dB      = 10.0 * np.log10(fs / N)               # offset in dB for an FFT bin frequency band

    sig_id      = np.argmax(amp_resp)                   # bin index of the signal
    sig_level   = amp_resp[sig_id] + bin_dB             # signal power level, dBFS

    est_flr     = np.mean(amp_resp)                     # average the amplitude response in dB, a guess of noise floor
    spur_id     = amp_resp > est_flr + 15               # find any spur 15 dB higher than the noise floor

    noise_bins  = PSD[np.invert(spur_id)]               # find all the noise bins (linear scale)
    noise_flr   = 10.0 * np.log10(np.mean(noise_bins))  # average the noise spectral density, dBFS/Hz

    noise_level = noise_flr + 10.0 * np.log10(fs/2)     # noise power level, dBFS
    snr         = sig_level - noise_level               # SNR, dB

    # collect the results
    result['freq']        = freq                        # Hz
    result['psd']         = PSD                         # linear PSD (power / Hz)
    result['amp_resp']    = amp_resp                    # dBFS/Hz
    result['pha_resp']    = pha_resp                    # degree
    result['signal_freq'] = freq[sig_id]                # signal frequency, Hz
    result['signal_mag']  = sig_level                   # signal magnitude, dBFS
    result['noise_flr']   = noise_flr                   # dBFS/Hz
    result['snr']         = snr                         # dB
    result['bin_db']      = bin_dB                      # dB
    result['status']      = True

    # make the plot
    if plot:
        from llrflibs.rf_plot import plot_calc_psd
        plot_calc_psd(result)
        
    # return
    return result

def calc_psd(data, fs, bit = 0, plot = False):
    '''
    Calculate the power spectral density of the input waveform (nominal sampling).
    
    Refer to LLRF Book section 6.1.2.

    Parameters:
        data:        numpy array, 1-D array of the raw waveform
        fs:          float, sampling frequency, Hz
        bit:         int, number of bit of data. If bit = 0, do not scale data
        plot:        boolean, True for plot the spectrum
        
    Returns:
        freq:        numpy array, frequency waveform, Hz
        amp_resp:    numpy array, amplitude response, dBFS/Hz
        pha_resp:    numpy array, phase response, degree
        signal_freq: float, signal frequency, Hz
        signal_mag:  float, signal level, dBFS
        noise_flr:   float, noise floor, dBFS/Hz
        snr:         float, signal-to-noise ratio, dB
        bin_db:      float, offset for an FFT bin freq space, dB
        enbw_db:     float, offset for an FFT bin freq space (correct windowing), dB
        status:      boolean, success (True) or fail (False)

    Note:
        The unit dBFS is for ADC raw data, if for phase in radian, it should be
        replaced by dBrad^2; if for relative amplitude, should be replaced by dB
    '''
    # results
    result = {'status': False}

    # get the input data size
    if isinstance(data, np.ndarray): 
        N = data.shape[0]
    elif isinstance(data, list):     
        N = len(data)
        data = np.array(data)
    else:
        N = 0

    # check the input
    if N < 3 or fs <= 0 or bit < 0:
        return result

    # calculate the FFT and get the results in Nyquist band
    if bit > 0:
        fftd = data[0:N] / (2.0**(bit-1)/np.sqrt(2.0))  # tailor the data and normalize with full scale
    else:
        fftd = data[0:N]

    win  = np.blackman(N)                               # blackman window    
    Y    = np.fft.fft(fftd * win)                       # do FFT on windowed data
    freq = np.arange(N) / N * fs                        # frequency vector, Hz
    ids  = freq <= 0.5 * fs                             # index of Nyquist band
    Y    = Y[ids]                                       # tailor the response
    freq = freq[ids]                                    # tailor the frequency vector

    Wn   = 1.0 / N * np.sum(win * win)                  # window correction factor
    Ws   = np.mean(win)**2 
    ENBW = Wn / Ws * fs / N                             # effective noise bandwidth for calculating the signal level

    # calculate the PSD
    PSD  = np.abs(Y) * np.abs(Y) / (N * fs * Wn)        # SSB spectra, linear scale relative power / Hz
    if (N % 2) == 0: PSD[1:-1] *= 2.0                   # convert to DSB spectra (exclude DC and f_nyquist)
    else:            PSD[1:]   *= 2.0                   # convert to DSB spectra (exclude DC)

    # calculate the derived quantities
    amp_resp    = 10.0 * np.log10(PSD)                  # in dBFS/Hz
    pha_resp    = np.angle(Y, deg = True)               # in degree
    bin_dB      = 10.0 * np.log10(fs / N)               # offset in dB for an FFT bin frequency band
    enbw_dB     = 10.0 * np.log10(ENBW)                 # offset in dB for an FFT bin frequency band (corrected the windowing)
                                                        # Note: "bin_dB" is used to integrate the noise power and the signal power
                                                        #       (using all signal bins including the leaked ones); "enbw_dB" is only
                                                        #       used in calcuating the signal level by taking only one central bin 
                                                        #       of the signal - to avoid integrating the spreaded signal bins (see
                                                        #       below for the "sig_level" calculation)

    sig_id      = np.argmax(amp_resp)                   # bin index of the signal
    sig_level   = amp_resp[sig_id] + enbw_dB            # signal power level (enbw_dB corrects the signal spectra leakage), dBFS
                                                        # Note: by this treatment, we obtain a "sig_level" equivalent integrating
                                                        #       all the signal bins that are formed by the leakage of the spectrum

    est_flr     = np.mean(amp_resp)                     # average the amplitude response in dB, a guess of noise floor
    spur_id     = amp_resp > est_flr + 15               # find any spur 15 dB higher than the noise floor

    noise_id    = np.invert(spur_id)                    # indices of noise bins
    noise_bins  = PSD[noise_id]                         # find all the noise bins (linear scale)    
    noise_flr   = 10.0 * np.log10(np.mean(noise_bins))  # average the noise spectral density, dBFS/Hz

    noise_level = noise_flr + 10.0 * np.log10(fs/2)     # noise power level, dBFS
    snr         = sig_level - noise_level               # SNR, dB

    # collect the results
    result['freq']        = freq                        # Hz
    result['psd']         = PSD                         # complex PSD
    result['amp_resp']    = amp_resp                    # dBFS/Hz
    result['pha_resp']    = pha_resp                    # degree
    result['signal_freq'] = freq[sig_id]                # signal frequency, Hz
    result['signal_mag']  = sig_level                   # signal magnitude, dBFS
    result['noise_flr']   = noise_flr                   # dBFS/Hz
    result['snr']         = snr                         # dB
    result['bin_db']      = bin_dB                      # dB
    result['enbw_db']     = enbw_dB                     # dB
    result['status']      = True

    # make the plot
    if plot:
        from llrflibs.rf_plot import plot_calc_psd
        plot_calc_psd(result)

    # return
    return result

def rand_unif(low = 0.0, high = 1.0, n = 1):
    '''
    produce random number within a certain range.
    
    Parameters:
        n:    int, number of output
        low:  float, low limit of the data
        high: float, high limit of the data
        
    Returns:
        val:  if n = 1, it is a float number, if n > 1, it is a np array
    '''
    # check the input
    if n < 1: n = 1

    # generate the random numbers
    val = np.random.uniform(low, high, n)

    # make return
    if n == 1: return val[0]
    else:      return val

def gen_noise_from_psd(freq_vector, pn_vector, fs, N):
    '''
    generate noise series from DSB PSD.

    Parameters:
        freq_vector: numpy array, offset frequency from carrier, Hz
        pn_vector:   numpy array, DSB noise PSD, dBrad^2/Hz for phase noise
        fs:          float, sampling frequency, Hz
        N:           float, number of samples
        
    Returns:
        status:      boolean, success (True) or fail (False)
        pn_series:   numpy array, time series of phase/amplitude noise
        freq_p:      numpy array, interpreted freq_vector input
        pn_p:        numpy array, interpreted pn_vector input
    '''
    # check input
    if (freq_vector.shape != pn_vector.shape) or (freq_vector.shape[0] < 2) or \
       (fs <= 0) or (N < 1):
        return False, None, None, None

    # prepare the frequency vector of the data series
    df_bin      = fs / N                                # frequency bin size, Hz
    freq        = np.arange(N) * df_bin                 # frequency vector, Hz
    freq[freq > fs/2] -= fs                             # full freq vector with negative freq, Hz
    
    # interpolate the PSD at positive frequencies
    freq_p      = freq[freq > 0]                        # +freq (excl. DC but with fs/2 if exists), Hz
    pn_p        = np.interp(10*np.log10(freq_p),        # linear interpolation in log scale
                            10*np.log10(freq_vector),
                            pn_vector)
    
    # calculate the spectrum (see eq (6.15) of LLRF book) and add random phases
    pn_p_cplx   = np.sqrt(10**(pn_p/10) * N * fs / 2) * \
                  np.exp(1j * rand_unif(low = -np.pi, high = np.pi, n = pn_p.shape[0]))
    
    # get the full complex spectrum (0 to fs) of the phase noise
    if np.mod(N, 2) == 0:
        pn_spec = np.concatenate([np.array([pn_p_cplx[0]]),         # fs/2 bin exists
                                  pn_p_cplx,                              
                                  np.conj(pn_p_cplx[-2::-1])])
    else:
        pn_spec = np.concatenate([np.array([pn_p_cplx[0]]),         # fs/2 bin does not exist
                                  pn_p_cplx,                              
                                  np.conj(pn_p_cplx[::-1])])

    # calculate the noise time series        
    pn_series = np.real(np.fft.ifft(pn_spec))
    pn_series -= np.mean(pn_series)
    
    return True, pn_series, freq_p, pn_p

def calc_rms_from_psd(freq_vector, pn_vector, freq_start, freq_end, fs, N):
    '''
    calculate the rms value from PSDs.

    Parameters:
        freq_vector: numpy array, offset frequency from carrier, Hz
        pn_vector:   numpy array, DSB noise PSD, dBrad^2/Hz for phase noise
        freq_start:  float, starting frequency for integration, Hz
        freq_end:    float, ending frequency for integration, Hz
        fs:          float, sampling frequency, Hz
        N:           int, number of points in the freq integration range
        
    Returns:
        status:      boolean, success (True) or fail (False)
        freq_p:      numpy array, interpreted freq_vector input
        pn_p:        numpy array, interpreted pn_vector input
        jitter_p:    numpy array, integrated jitter starting from freq_start
                      to different freq_p element values
    '''
    # check input
    if (freq_vector.shape != pn_vector.shape) or (freq_vector.shape[0] < 2) or \
       (fs <= 0) or (N <= 0) or (freq_start <= 0) or (freq_end <= freq_start):
        return False, None, None, None
            
    # get the frequeny vector in the integration range and interpolate the PSD
    df_bin  = (freq_end - freq_start) / N
    freq_p  = np.linspace(freq_start, freq_end, N + 1)
    pn_p    = np.interp(10*np.log10(freq_p),            # linear interpolation in log scale
                        10*np.log10(freq_vector),
                        pn_vector)
    
    # integrate the jitter in the frequency integration region
    bin_dB      = 10.0 * np.log10(df_bin)
    jitter_p    = np.zeros(freq_p.shape)    
    noise_power = 0.0
    
    for i in range(N+1):
        jitter_p[i] = np.sqrt(noise_power)
        noise_power += 10.0**((pn_p[i] + bin_dB) / 10.0)    
    
    return True, freq_p, pn_p, jitter_p
        
def notch_filt(wf, fnotch, Q, fs, b = None, a = None):
    '''
    Apply notch filter to the signal.

    Parameters:
        wf:      numpy array, the waveform to be notch filtered
        fnotch:  float, frequency to be notched, Hz
        Q:       float, quality factor of the notch filter
        fs:      float, sampling frequency, Hz
        b, a:    numpy arrays filter coefficients. Note that the user can
                  either input ``b, a``, or ``fnotch, Q, fs``
                  
    Returns:
        status:  boolean, success (True) or fail (False)
        wf_f:    numpy array, filtered waveform
        b, a:    numpy array, filter coefficients. If the same filter 
                  is used to filter multiple waveforms, we can use it
                  to avoid repeat the filter synthesis
    '''
    # check the input
    if (wf is None) or (wf.shape[0] < 3) or (fnotch <= 0.0) or \
       (Q <= 0.0) or (fs <= 0.0):
        return False, None, None, None

    # create the notch filter if not available
    if (b is None) or (a is None):
        b, a = signal.iirnotch(fnotch, Q, fs)

    # filter the data
    return True, signal.filtfilt(b, a, wf), b, a

def design_notch_filter(fnotch, Q, fs):
    '''
    Design the notch filter (return a discrete filter).

    Parameters:
        fnotch:  float, frequency to be notched, Hz
        Q:       float, quality factor of the notch filter
        fs:      float, sampling frequency, Hz
        
    Returns:
        status:         boolean, success (True) or fail (False)
        Ad, Bd, Cd, Dd: numpy matrix, discrete notch filter
    '''
    # check the input
    if (fnotch <= 0.0) or (Q <= 0.0) or (fs <= 0.0):
        return (False,) + (None,)*4    

    # create the notch filter
    b, a = signal.iirnotch(fnotch, Q, fs)
    dsys = signal.dlti(b, a, dt = 1.0/fs)
    dsys = signal.StateSpace(dsys)
    Ad, Bd, Cd, Dd, dt = dsys.A, dsys.B, dsys.C, dsys.D, dsys.dt    

    # filter the data
    return True, Ad, Bd, Cd, Dd

def filt_step(Afd, Bfd, Cfd, Dfd, in_step, state_f0):
    '''
    Apply a step of the filter.

    Parameters:
        Afd, Bfd, Cfd, Dfd: numpy matrix, discrete filter model
        in_step:            float or complex, input of the time step
        state_f0:           numpy matrix, state of the filter of last step
        
    Returns:
        status:   boolean, success (True) or fail (False)
        out_step: float or complex, filter output of this time step
        state_f:  numpy matrix, state of the filter of this step, 
                   should input to the next execution    
    '''
    # calculate the controller output
    state_f  = Afd * state_f0 + Bfd * in_step
    out_step = Cfd * state_f0 + Dfd * in_step

    # return the results of the step
    return True, out_step, state_f

def moving_avg(wf_in, n):
    '''
    Moving average without compensating the group delay.
    
    Parameters:
        wf_in:   numpy array, input waveform
        n:       int, point number of moving average
        
    Returns:
        status:  boolean, success (True) or fail (False)
        wf_out:  numpy array, output waveform
    '''
    # check the input
    if n <= 0 or wf_in.shape[0] < 3:
        return False, None

    # make the moving average
    wf_out     = np.cumsum(wf_in, dtype = float)
    wf_out[n:] = (wf_out[n:] - wf_out[:-n]) / n
    wf_out[:n] = wf_in[:n]

    return True, wf_out
    
def rand_sine(N, fs, nfreq = 1, Amin = 0.0, Amax = 1.0, fmin = 0.0, fmax = 1e3):
    '''
    Generate a data series of the sum of several random sine signals.
    
    Parameters:
        N:          int, number of point of the series
        fs:         float, sampling frequency, Hz
        nfreq:      int, number of frequencies
        Amin, Amax: float, min and max values of amplitude
        fmin, fmax: float, min and max values of frequencies, Hz
        
    Returns:
        status:     boolean, success (True) or fail (False)
        sout:       numpy array, output data series
        t:          numpy array, time series, s
    '''
    # check the inputs
    if N <= 0 or fs <= 0 or nfreq < 1 or Amin < 0 or \
       Amax < Amin or fmin < 0 or fmax < fmin:
        return False, None
        
    # generate the random amplitude, phase and frequency
    A = np.random.uniform(Amin, Amax, nfreq)
    f = np.random.uniform(fmin, fmax, nfreq)
    P = np.random.uniform(-np.pi, np.pi, nfreq)

    # generate the series
    sout = np.zeros(N)
    t    = np.arange(N) / fs
    for i in range(len(A)):
        sout = sout + A[i] * np.sin(2.0 * np.pi * f[i] * t + P[i])

    # return
    return True, sout, t















