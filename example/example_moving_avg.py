###################################################################################
#  Copyright (c) 2025 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
###################################################################################
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example code to demonstrate moving average
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import numpy as np
import matplotlib.pyplot as plt

from llrflibs.rf_noise import *

# a test waveform
data = np.zeros(2048)
data[100:1500] = 1.0
data += rand_unif(low = -0.1, high = 0.1, n = 2048)

# make average
status1, data_f1 = moving_avg(data, 5)
status2, data_f2 = moving_avg(data, 8)
#status1, data_f1 = moving_avg_obs(data, 5)     # moving_avg_obs has no group delay compensation, should not be used any longer
#status2, data_f2 = moving_avg_obs(data, 8)

plt.figure()
plt.plot(data, label = 'Initial waveform')
plt.plot(data_f1, label = 'Moving avg: n = 5')
plt.plot(data_f2, label = 'Moving avg: n = 8')
plt.grid()
plt.legend()
plt.show(block = False)



