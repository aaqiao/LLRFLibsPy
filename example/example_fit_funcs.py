###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to fit functions:
    - sine
    - cosine
    - circle
    - ellipse
    - Guassian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_fit import *
from rf_misc import *

# parameters
pi = np.pi          # shorter pi

# ------------------------------------------------------
# define a sine/cosine function
# ------------------------------------------------------
X  = np.linspace(-4*pi, 4*pi, 1000)             # phase as indepedent variable, rad
Y1 = 2.05  * np.cos(X - pi/5) - 1.24            # test cosine function
Y2 = 33.44 * np.sin(X - pi/4) - 2.88            # test sine function

status, A1, phi1, c1 = fit_sincos(X, Y1, 'cos') # fit the cosine function
status, A2, phi2, c2 = fit_sincos(X, Y2, 'sin') # fit the sine function

print(A1, phi1, c1, A2, phi2, c2)               # compare with the given values

# ------------------------------------------------------
# define a circle
# ------------------------------------------------------
phi_data   = np.linspace(-pi, pi, 20)           # for generating the data points
phi_verify = np.linspace(-pi, pi, 1000)         # for verifying the fitting
A   = 3.44                                      # radius of the circle
X   = A * np.cos(phi_data) + 5.3 + np.random.normal(0.0, A/10, phi_data.shape)  # data with noise
Y   = A * np.sin(phi_data) - 4.4 + np.random.normal(0.0, A/10, phi_data.shape)

status, x0, y0, r = fit_circle(X, Y)            # fit the circle

plt.figure()
plt.plot(X, Y, '*', label = 'Data Points')
plt.plot(x0 + r * np.cos(phi_verify), y0 + r * np.sin(phi_verify), label = 'Fitted Circle')
plt.grid()
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show(block = False)

# ------------------------------------------------------
# define an ellipse (reuse some definition in circle)
# ------------------------------------------------------
a0    = 5                                                           # semi-major of demo ellipse
b0    = 2                                                           # semi-minor of demo ellipse
sita0 = 30 * np.pi / 180                                            # tilt angle of demo ellipse

X1    = a0 * np.cos(phi_data)                                       # define the basic ellipse (can also use "plot_ellipse" function)
Y1    = b0 * np.sin(phi_data)
X2    = X1 * np.cos(sita0) - Y1 * np.sin(sita0)                     # rotation
Y2    = X1 * np.sin(sita0) + Y1 * np.cos(sita0)
X     = X2 + 5.3 + np.random.normal(0.0, a0/20, phi_data.shape)     # shift + noise
Y     = Y2 - 4.4 + np.random.normal(0.0, a0/20, phi_data.shape)

_, [A,B,C,D,E,F], a, b, x0, y0, sita = fit_ellipse(X, Y)            # fit the ellipse

X_f  = np.linspace(0,10,100)                                        # plot using fitted general equation
Y_f1 = (-(B*X_f+E) + np.sqrt((B*X_f+E)**2 - 4*(A*X_f**2+D*X_f+F))) / 2.0
Y_f2 = (-(B*X_f+E) - np.sqrt((B*X_f+E)**2 - 4*(A*X_f**2+D*X_f+F))) / 2.0

_, X3_f, Y3_f = plot_ellipse(1000, a    = a, 
                                   b    = b, 
                                   x0   = x0, 
                                   y0   = y0, 
                                   sita = sita)                     # plot using fitted characteristics

plt.figure()
plt.plot(X, Y, '*', label = 'Data Points')
plt.plot(np.hstack((X_f,X_f)), np.hstack((Y_f1, Y_f2)), '-o', label = 'Fitting (general)')
plt.plot(X3_f, Y3_f, '-', label = 'Fitting (charateristics)')
plt.grid()
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show(block = False)

# ------------------------------------------------------
# test Guassian fit
# ------------------------------------------------------
# generate a Guassian noise
mu0  = 2.3                                                          # mean value
var  = 11.0                                                         # variance (= sigma^2)
data = np.random.normal(mu0, np.sqrt(var), 10000)                   # get random noise data

# make a statistics of the histogram
n = 100
Y, bin_edges = np.histogram(data, bins = n)                         # make statistics
X = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(n)])

status, a, mu, sigma = fit_Gaussian(X, Y)                           # fit Guassian function

status, X_f, Y_f = plot_Guassian(n, a = a, mu = mu, sigma = sigma)

plt.figure()
plt.plot(X, Y, '*', label = 'Data Points')
plt.plot(X_f, Y_f, '-', label = 'Fitting')
plt.grid()
plt.legend()
plt.xlabel('X')
plt.ylabel('Probability (not normalized)')
plt.show(block = False)












