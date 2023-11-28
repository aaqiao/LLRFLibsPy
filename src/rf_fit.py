"""Fit data to sine/cosine, circle, ellipse or Gaussian functions."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for fitting algorithms

Implemented:
    - fit_sincos    : fit the sine or cosine function
    - fit_circle    : fit the 2D points to a circle function
    - fit_ellipse   : fit the 2D points to an ellipse function
    - fit_Gaussian  : fit a 1D Guassian function
#########################################################################
'''
import numpy as np
from scipy.optimize import curve_fit

def fit_sincos(X_rad, Y, target = 'cos'):
    '''
    Fit the sine or cosine function
      - ``y = A * cos(x + phi) + c = (A*cos(phi)) * cos(x) - (A*sin(phi)) * sin(x) + c``
      - ``y = A * sin(x + phi) + c = (A*cos(phi)) * sin(x) + (A*sin(phi)) * cos(x) + c``

    Parameters:
        X_rad:   numpy array, phase array in radian
        Y:       numpy array, value of the sine or cosine function
        target:  'sin' or 'cos', determine which function to fit to
        
    Returns:
        status:  boolean, success (True) or fail (False)
        A:       float, amplitude of the function
        phi_rad: float, phase of the function, rad
        c:       float, offset of the function
    '''
    # check the input
    if (not X_rad.shape == Y.shape) or (X_rad.shape[0] < 3):
        return False, None, None, None

    # make the least-square fitting
    if target == 'cos':
        P = np.linalg.lstsq(np.vstack((np.cos(X_rad), -np.sin(X_rad), np.ones(Y.shape))).T, Y, rcond = None)
    else:
        P = np.linalg.lstsq(np.vstack((np.sin(X_rad),  np.cos(X_rad), np.ones(Y.shape))).T, Y, rcond = None)

    # calculate the result
    A       = np.sqrt(P[0][0]**2 + P[0][1]**2)    # P[0][0] = A*cos(phi); P[0][1] = A*sin(phi)
    phi_rad = np.arctan2(P[0][1], P[0][0])
    c       = P[0][2]

    return True, A, phi_rad, c

def fit_circle(X, Y):
    '''
    Fit a circle. Given ``x``, ``y`` data points and find ``x0``, ``y0`` and ``r`` following
    ``(x - x0)^2 + (y - y0)^2 = r^2``.

    Parameters:
        X:      numpy array, x-coordinate of the points
        Y:      numpy array, y-coordinate of the points
        
    Returns:
        status: boolean, success (True) or fail (False)
        x0, y0: float, center coordinate of the circle
        r:      float, radius of the circle
    '''
    # check the input
    if (not X.shape == Y.shape) or (X.shape[0] < 3):
        return False, None, None, None

    # make the fit again with least square
    P = np.linalg.lstsq(np.vstack((2*X, 2*Y, np.ones(Y.shape))).T, X**2+Y**2, rcond = None)

    # calculate the result
    x0 = P[0][0]
    y0 = P[0][1]
    r  = np.sqrt(P[0][2] + x0**2 + y0**2)

    return True, x0, y0, r

def fit_ellipse(X, Y):
    '''
    Fit an ellipse. Get the general ellipse function and its characteristics, including
    semi-major axis ``a``, semi-minor axis ``b``, center coordinates ``(x0, y0)`` and rotation
    angle ``sita`` (the angle from the positive horizontal axis to the ellipse's major axis).
    The general ellipse equation is ``A*X^2 + B*X*Y + C*Y^2 + D*X + E*Y + F = 0``.
    When making the fitting, we divide the equation by ``C`` to normalize the coefficient of 
    ``Y^2`` to 1 and move it to the right side of the fitting equation. The ellipse is derived 
    with the following steps:
    
     1. define a canonical ellipse: 
            - ``X1 = a * cos(phi)``
            - ``Y1 = b * sin(phi)`` 
            
        where ``phi`` is a phase vector covering from 0 to 2*pi
        
     2. rotate the canonical ellipse: 
            - ``X2 = X1 * cos(sita) - Y1 * sin(sita)``
            - ``Y2 = X1 * sin(sita) + Y1 * cos(sita)``
            
     3. add offset to the rotated ellipse:
            - ``X = X2 + x0``
            - ``Y = Y2 + y0``

    See the webpage: https://en.wikipedia.org/wiki/Ellipse.  

    Parameters:
        X:      numpy array, x-coordinate of the points
        Y:      numpy array, y-coordinate of the points
        
    Returns:
        status: boolean, success (True) or fail (False)
        Coef:   list, coefficiets derived from the least-square fitting
        a:      float, semi-major
        b:      float, semi-minor
        x0, y0: float, center of the ellipse
        sita:   float, angle of the ellipse (see above), rad
    '''
    # check the input
    if (not X.shape == Y.shape) or (X.shape[0] < 3):
        return (False,) + (None,)*6

    # make the fit with least square
    P = np.linalg.lstsq(np.vstack((X**2, X*Y, X, Y, np.ones(Y.shape))).T, -Y**2, rcond = None)

    # calculate the characteristics
    A = P[0][0]
    B = P[0][1]
    C = 1.0
    D = P[0][2]
    E = P[0][3]
    F = P[0][4]

    a  = -np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C) + np.sqrt((A-C)**2 + B**2))) / (B**2 - 4*A*C)
    b  = -np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F)*((A+C) - np.sqrt((A-C)**2 + B**2))) / (B**2 - 4*A*C)
    x0 = (2*C*D - B*E) / (B**2 - 4*A*C)
    y0 = (2*A*E - B*D) / (B**2 - 4*A*C)

    if B != 0.0:
        sita = np.arctan((C - A - np.sqrt((A-C)**2 + B**2)) / B)
    else:
        if A <= C:
            sita = 0
        else:
            sita = np.pi/2

    return True, [A,B,C,D,E,F], a, b, x0, y0, sita

def fit_Gaussian(X, Y):
    '''
    Fit a Guassian distribution function. 
    
    See the webpage: https://pythonguides.com/scipy-normal-distribution/.

    Parameters:
        X:      numpy array, x-coordinate of the points
        Y:      numpy array, y-coordinate of the points
        
    Returns:
        status: boolean, success (True) or fail (False)
        a:      float, magnitude scale of the un-normalized distribution
        mu:     float, mean value
        sigma:  float, standard deviation
    '''
    # check the input
    if (not X.shape == Y.shape) or (X.shape[0] < 3):
        return (False,) + (None,)*3
    
    # define the Gaussian (normal) distribution 
    def norm_dist(x, a, mu, var):
        y = a * np.exp(-0.5 / var * (x - mu)**2)    # since X,Y are not normalized, "a" can be arbitrary value
        return y

    # get the initial guess of the parameters
    mu_est  = sum(X * Y) / sum(Y)
    var_est = sum((X - mu_est)**2 * Y) / sum(Y)

    # make the fit
    P, cov  = curve_fit(norm_dist, X, Y, p0 = [1.0, mu_est, var_est])
    a, mu, var = P[0], P[1], P[2]

    return True, a, mu, np.sqrt(var)



















