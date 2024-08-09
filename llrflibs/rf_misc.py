"""Save/read data to/from Matlab files."""
#############################################################################
#  Copyright (c) 2023 by Paul Scherrer Institute, Switzerland
#  All rights reserved.
#  Authors: Zheqiao Geng
#############################################################################
'''
#########################################################################
Here collects routines for variaous functions
 
Implemented:
    - save_mat        : save a dictionary into a Matlab .mat file
    - load_mat        : load a Matlab .mat file into a dictionary
    - get_curtime_str : get a string of the current date/time
    - get_bit         : get a bit of an integer
    - add_tf          : adding two transfer function in num/den format
    - plot_ellipse    : plot an ellipse using its characteristics
    - plot_Guassian   : plot a 1D Guassian distribution
#########################################################################
'''
import datetime
import numpy as np
import scipy.io as spio

def save_mat(data_dict, file_name):
    '''
    Save a dictionary into matlab file.

    Parameters:
        data_dict: data dictionary
        file_name: full file name including path
        
    Returns:
        status:    boolean, success (True) or fail (False)
    '''
    # check input
    if not isinstance(data_dict, dict):
        return False

    # array dict
    data = {}

    # convert the list into array
    for item in data_dict.keys():
        item_matlab_str = item.replace("-", "_")    # convert names suitable for matlab
        item_matlab_str = item_matlab_str.replace(":", "_")

        if isinstance(data_dict[item], list):
            data[item_matlab_str] = np.array(data_dict[item])
        else:
            data[item_matlab_str] = data_dict[item]

    # check the extention of the file_name
    if not '.mat' in file_name:
        file_name += '.mat'

    # write to matlab file
    try:
        spio.savemat(file_name        = file_name,
                     mdict            = data,
                     long_field_names = True, 
                     oned_as          = "column")
        return True
    except:
        return False

def load_mat(file_name, to_list = False):
    '''
    Load a matlab data file into a dict. this function should be called instead 
    of direct spio.loadmat as it cures the problem of not properly recovering 
    python dictionaries from mat files. It calls the function check keys to cure 
    all entries which are still mat-objects.

    Parameters:
        file_name:  full file name including path
        
    Returns:
        status:    boolean, success (True) or fail (False)
        data:      dict, contain the data with the key the var name in Matlab

    Note:
       This implementation can be found here
       https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries.
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                if to_list:
                    d[strg] = _tolist(elem)
                else:
                    d[strg] = elem
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(file_name, 
                        struct_as_record = False, 
                        squeeze_me       = True, 
                        mat_dtype        = False)
    return _check_keys(data)

def get_curtime_str(format = '%Y-%m-%d %H:%M:%S'):
    '''
    Get the current time string.
    
    Parameters:
        format:   string of format
        
    Returns:
        time_str: string, the time string
    '''
    cur_time = datetime.datetime.now()
    return cur_time.strftime(format)

def get_bit(data, bit_id = 0):
    '''
    Get a bit from the data, the bit index starts from 0.
    
    Parameters:
        data:   int, the input data
        bit_id: int, index of the bit
        
    Returns: 
        bit:    int, 1 or 0
    '''
    # convert the data and bit id to integer
    data_i = int(data)
    id_i   = int(bit_id)

    # return the bit
    return ((data_i >> id_i) & 1)

def add_tf(num1, den1, num2, den2):
    '''
    Add two transfer functions in num/den format ``A/B+C/D = (AD+BC)/BD``.

    Parameters:
        num1, den1:  list, polynomial coeffcients of transfer function1
        num2, den2:  list, polynomial coeffcients of transfer function2
        
    Returns:
        num_sum, den_sum: list, polynomial coefficients of the sum
    '''
    # convert to polynomial object
    A = np.poly1d(num1)
    B = np.poly1d(den1)
    C = np.poly1d(num2)
    D = np.poly1d(den2)

    # get the result
    num_sum = A * D + B * C
    den_sum = B * D
    return num_sum.c.tolist(), den_sum.c.tolist()

def plot_ellipse(n, a = 1.0, b = 1.0, x0 = 0.0, y0 = 0.0, sita = 0.0, plot = False):
    '''
    Draw an ellipse (see ``fit_ellipse`` function in ``rf_fit`` module).

    Parameters:
        n:      int, number of points
        a:      float, semi-major
        b:      float, semi-minor
        x0, y0: float, center of the ellipse
        sita:   float, angle of the ellipse, rad
        plot:   boolean, True for enabling displaying
        
    Returns:
        status: boolean, True for success
        X, Y:   numpy array, points on the ellipse
    '''
    # check the input
    if (n < 3) or (a <= 0.0) or (b <= 0.0):
        return False, None, None

    # generate points on the ellipse
    phi = np.linspace(-np.pi, np.pi, n)

    X1 = a * np.cos(phi)                        # a standard ellipse
    Y1 = b * np.sin(phi)
    X2 = X1 * np.cos(sita) - Y1 * np.sin(sita)  # make rotation
    Y2 = X1 * np.sin(sita) + Y1 * np.cos(sita)
    X  = X2 + x0                                # add offset
    Y  = Y2 + y0

    # plot the ellipse
    if plot:
        from llrflibs.rf_plot import plot_plot_ellipse
        plot_plot_ellipse(X, Y)

    # return the points
    return True, X, Y

def plot_Guassian(n, a = 1.0, mu = 0.0, sigma = 1.0, plot = False):
    '''
    Draw Guassiand distribution (see ``fit_Guassian`` function in ``rf_fit`` module).

    Parameters:
        n:     int, number of points
        a:     float, magnitude scale factor
        mu:    float, mean value
        sigma: float, standard deviation
        plot:  boolean, True for enabling displaying
        
    Returns:
        status: boolean, True for success
        X, Y:   numpy array, points on the curve
    '''
    # check the input
    if (n < 3) or (a <= 0.0) or (sigma <= 0.0):
        return False, None, None

    # generate the data of 5 sigma
    X = np.linspace(-5.0 * sigma + mu, 5.0 * sigma + mu, n)
    Y = a * np.exp(-0.5 / sigma**2 * (X - mu)**2)

    # plot
    if plot:
        from llrflibs.rf_plot import plot_plot_Guassian
        plot_plot_Guassian(X, Y)

    # return the results
    return True, X, Y



















