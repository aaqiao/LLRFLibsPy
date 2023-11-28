"""Plotting functions for internal use."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects all plotting functions. The goal is to avoid importing the 
"matplotlib" module in the algorithm codes to enable running the algorithms
in an embedded CPU where the "matplotlib" is not installed
#########################################################################
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_ss_discrete(fc, Ac_dB, Pc_deg, fd, Ad_dB, Pd_deg, Ts):
    '''
    Plot the frequency responses of the continous and discrete state-space
    equations, used in function ``ss_discrete`` of the ``rf_control`` module.
    '''
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(fc, Ac_dB, label = 'Continous')
    plt.plot(fd, Ad_dB, label = 'Discrete')
    plt.axvline( 0.5 / Ts, ls = '--')
    plt.axvline(-0.5 / Ts, ls = '--')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2,1,2)
    plt.plot(fc, Pc_deg, label = 'Continous')
    plt.plot(fd, Pd_deg, label = 'Discrete')
    plt.axvline( 0.5 / Ts, ls = '--')
    plt.axvline(-0.5 / Ts, ls = '--')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    plt.suptitle('State-space System Discretization')
    plt.show(block = False)
    
def plot_ss_freqresp(f_wf, A_wf_dB, P_wf_deg, h, fs, Ts, title):
    '''
    Plot the frequency response of a state-space system, used in function 
    ``ss_freqresp`` of the ``rf_control`` module.
    '''
    plt.figure()
    plt.subplot(2,2,1)                          # bode plot
    plt.plot(f_wf, A_wf_dB)
    if Ts is not None:
        plt.axvline( fs / 2, ls = '--')
        plt.axvline(-fs / 2, ls = '--')
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2,2,3)
    plt.plot(f_wf, P_wf_deg)
    if Ts is not None:
        plt.axvline( fs / 2, ls = '--')
        plt.axvline(-fs / 2, ls = '--')
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    plt.subplot(1,2,2)                          # Nyquist plot (need to plot +/- frequencies differently)
    plt.plot(np.real(h[f_wf >= 0.0]), np.imag(h[f_wf >= 0.0]), label = 'Positive Frequency')
    plt.plot(np.real(h[f_wf <  0.0]), np.imag(h[f_wf <  0.0]), label = 'Negative Frequency')
    plt.plot([-1], [0], '*')
    plt.legend()
    plt.grid()
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.suptitle(title)
    plt.show(block = False)
    
def plot_basic_rf_controller(w, h):
    '''
    Plot the frequency response of the basic RF controller, used in function 
    ``basic_rf_controller`` of the ``rf_control`` module.
    '''
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w / 2 / np.pi, 20*np.log10(np.abs(h)), label = 'Controller')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2,1,2)
    plt.plot(w / 2 / np.pi, np.angle(h, deg = True), label = 'Controller')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    plt.suptitle('Basic RF Controller')
    plt.show(block = False)

def plot_loop_analysis(f_wf, L, S, T, fs, Ts, label):
    '''
    Plot the loop analysis results, used in function ``loop_analysis`` of the 
    ``rf_control`` module.
    '''
    plt.figure()
    plt.subplot(2,2,1)                          # bode plot
    plt.plot(f_wf, 20*np.log10(np.abs(L)),       label = 'L')
    plt.plot(f_wf, 20*np.log10(np.abs(S)), '-.', label = 'S')
    plt.plot(f_wf, 20*np.log10(np.abs(T)), ':',  label = 'T')
    if Ts is not None:
        plt.axvline( fs / 2, ls = '--')
        plt.axvline(-fs / 2, ls = '--')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Amplitude Response of L/S/T')

    plt.subplot(2,2,3)
    plt.plot(f_wf, np.angle(L, deg = True),       label = 'L')
    plt.plot(f_wf, np.angle(S, deg = True), '-.', label = 'S')
    plt.plot(f_wf, np.angle(T, deg = True), ':',  label = 'T')
    if Ts is not None:
        plt.axvline( fs / 2, ls = '--')
        plt.axvline(-fs / 2, ls = '--')
    plt.axhline( 180, ls = '--')
    plt.axhline(-180, ls = '--')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    plt.title('Phase Response of L/S/T')

    plt.subplot(1,2,2)                          # Nyquist plot (need to plot +/- frequencies differently)
    plt.plot(np.real(L[f_wf >= 0.0]), np.imag(L[f_wf >= 0.0]), label = 'Positive Frequency')
    plt.plot(np.real(L[f_wf <  0.0]), np.imag(L[f_wf <  0.0]), label = 'Negative Frequency')
    plt.plot([-1], [0], '*')
    plt.legend()
    plt.grid()
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title('Nyquist Plot of L')

    plt.suptitle('Feedback Loop Analysis ' + label)
    plt.show(block = False)
    
def plot_calc_psd(result):
    '''
    Plot the spectrum, used in function ``calc_psd_coherent`` and ``calc_psd`` of the 
    ``rf_noise`` module.
    '''
    plt.figure()
    plt.plot(result['freq'], result['amp_resp'])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dBFS/Hz)')
    plt.grid()
    plt.show(block = False)        

def plot_cav_ss(wrf, hrf, wbm, hbm):
    '''
    Plot the spectra of cavity model, used in function ``cav_ss`` of the 
    ``rf_sim`` module.
    '''
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(wrf / 2 / np.pi, 20*np.log10(np.abs(hrf)), label = 'Cavity resp. to RF drive')
    plt.plot(wbm / 2 / np.pi, 20*np.log10(np.abs(hbm)), label = 'Cavity resp. to beam drive')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2,1,2)
    plt.plot(wrf / 2 / np.pi, np.angle(hrf, deg = True), label = 'Cavity resp. to RF drive')
    plt.plot(wbm / 2 / np.pi, np.angle(hbm, deg = True), label = 'Cavity resp. to beam drive')
    plt.legend()
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    plt.suptitle('Cavity State-space Model')
    plt.show(block = False)

def plot_rf_power_req(Pfor, Pref, QL_vec):
    '''
    Plot the required RF power, used in function ``rf_power_req`` of the 
    ``rf_sim`` module.
    '''
    plt.figure()
    plt.subplot(1,2,1)
    for dw in Pfor.keys():
        plt.plot(QL_vec, Pfor[dw] / 1000.0, 
                 label = 'Detuning = {:.1f} Hz'.format(dw/2/np.pi))
    plt.legend()
    plt.grid()
    plt.xlabel(r'$Q_L$')
    plt.ylabel(r'$P_{for}$ (kW)')
    plt.subplot(1,2,2)
    for dw in Pfor.keys():
        plt.plot(QL_vec, Pref[dw] / 1000.0, 
                 label = 'Detuning = {:.1f} Hz'.format(dw/2/np.pi))
    plt.legend()
    plt.grid()
    plt.xlabel(r'$Q_L$')
    plt.ylabel(r'$P_{ref}$ (kW)')
    plt.show(block = False)

def plot_plot_ellipse(X, Y):
    '''
    Plot the ellipse, used in function ``plot_ellipse`` of the ``rf_misc`` module.
    '''
    plt.figure()
    plt.plot(X, Y, '-*')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.axis('equal')
    plt.show(block = False)

def plot_plot_Guassian(X, Y):
    '''
    Plot the Guassian distribution, used in function ``plot_Guassian`` of the 
    ``rf_misc`` module.
    '''
    plt.figure()
    plt.plot(X, Y, '-*')
    plt.xlabel('X')
    plt.ylabel('Probability (not normalized)')
    plt.grid()
    plt.show(block = False)












    