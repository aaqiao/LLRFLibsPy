###################################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code for response matrix inversion, two methods are demonstrated
    - matrix inversion with SVD (with singular value filtering)
    - matrix inversion with least square method (with regularization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_control import *

# define the response matrix (see Eq. 2.7 of intelligent beam control book)
R = np.matrix([[ 0.338485346238024,   0.106882042994840,                   0,                   0,                   0,                   0,                   0,                   0],
               [ 0.877327812178906,  -0.213330054364056,   1.541853664555557,  -0.107419471256316,  -1.188259953276194,  -0.148821427513276,                   0,                   0],
               [-0.259536009211190,  -0.775792876838995,  -0.464579654860511,  -0.201156242742890,  -0.034972589648221,   0.528936256575156,                   0,                   0],
               [-0.661113748105799,   0.206844458447236,  -1.590213133773256,   0.082709324734732,   1.206499859270068,  -0.074493709357442,  -1.122814414271664,   3.044016551815725],
               [ 0.399705982986835,  -0.274205923575642,   0.723345231804639,  -0.047037480892735,  -0.785241582814755,   0.229474556653552,   0.856925940591046,  -0.546347202469690]])

n_in  = R.shape[1]                                              # number of inputs
n_out = R.shape[0]                                              # number of outputs
_,S,_ = np.linalg.svd(R)                                        # singular values of the response matrix

print("R's condition number is {:.2f}".format(np.linalg.cond(R)))
print("R's singular values are ", S)

# inverse the matrix
#Rinv = resp_inv_svd(R, singular_val_filt = 0.0)                 # you can change the filt threshold to see results
Rinv = resp_inv_lsm(R, regu = 0.1)                              # you can change the regularization (> 0) to see results

# simulate the closed loop
r       = np.matrix([[0.1, 0.2, 0.3, 0.4, 0.5]]).T              # setpoint
u       = np.matrix(np.zeros((n_in, 1)))                        # input vector
gain    = 0.1                                                   # feedback gain (discrete integral control)
n_iter  = 5000                                                  # iterations of simulation
noise   = 0.02                                                  # noise added to measurement
dR      = 1.2 * np.matrix(np.random.rand(n_out, n_in))          # response matrix perturbation (you can change the magnitude)

U       = np.zeros((n_iter, n_in))                              # collect all input data
Y       = np.zeros((n_iter, n_out))                             # collect all output data

for i in range(n_iter):
    # simulate the system output
    y = (R + dR) * u + noise * np.random.rand(u.shape[1], 1)    # emulate the perturbed noisy plant to be controlled

    # feedback for one step
    u = u + gain * Rinv * (r - y)

    # store the data
    U[i] = u.flatten()
    Y[i] = y.flatten()

# diplay
plt.figure(figsize = (12, 5))
plt.subplot(1,2,1)
for i in range(n_out):
    plt.plot(Y[:, i], label = 'Output {}'.format(i+1))
plt.grid()
plt.legend()
plt.xlabel('Simulation Step')
plt.ylabel('Output')
plt.subplot(1,2,2)
for i in range(n_in):
    plt.plot(U[:, i], label = 'Input {}'.format(i+1))
plt.grid()
plt.legend()
plt.xlabel('Simulation Step')
plt.ylabel('Input')
plt.show(block = False)







