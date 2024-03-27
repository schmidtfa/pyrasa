from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


def gen_log_space(limit, n):

    '''Taken from online RECHECK IF ITS ACTUALLY CORRECT'''

    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)



def lin_reg(x, a, b):

    '''Linear fit w/o knee NOTE: Data should be in loglog coordinates before running this'''

    y_hat = a + b * x

    return y_hat


def piecewise_linear(x, x1, b, k1, k2):

    '''Knee Fit using a piecewise linear regression NOTE: Data should be in loglog coordinates before running this'''

    condlist = [x < x1,  x > x1]
    funclist = [lambda x: k1*x + b, lambda x: k1*x + b + k2*(x-x1)]
    return np.piecewise(x, condlist, funclist)


def expo_reg(x, a, k, b):

    '''Uses specparams fitting function with a knee'''

    y_hat = a - np.log10(k + x**b)

    return y_hat


def expo_reg_nk(x, a, b):

    '''Uses specparams fitting function without a knee'''

    y_hat = a - np.log10(x**b)

    return y_hat


def compute_slope_expo(freq, psd, fit_func):

    if fit_func == 'knee':
        fit_f = expo_reg
    else:
        fit_f = expo_reg_nk
    
    p, _ = curve_fit(fit_f, freq, np.log10(psd))

    if fit_func == 'knee':
        #sometimes the knee frequency is unreasonably large (i.e. outside the fitting range)
        #take this as a marker that model fitting went wrong and return nans
        if 10**p[0] > freq.max():
            p = p * np.nan

        params = pd.DataFrame({'Knee': p[1],
                               'Knee Frequency (Hz)': p[1] ** (1. / p[2]),
                               'Offset': p[0],
                               'Exponent': p[2],
                              }, index=[0])
    else:
        params = pd.DataFrame({'Offset': p[0],
                              'Exponent': p[1]}, 
                               index=[0])


    return params


def compute_slope(freq, psd, fit_func, interp_factor=1000, n_log_vals=100):

    '''Compute the slope of the aperiodic spectrum'''

    if fit_func == 'linear':
        fit_f = lin_reg
    elif fit_func == 'knee':
        fit_f = piecewise_linear
    else:
        print('Fit function has to be either "linear" or "knee"')

    #need to interpolate as often less points in the beginning
    f_interp = interp1d(np.log10(freq),np.log10(psd), kind='linear')
    freq_sim = np.log10(np.linspace(freq.min(), freq.max(), num=len(freq)*interp_factor))
    psd_sim = f_interp(freq_sim)

    #we need equally distant points in log-log
    idcs = gen_log_space(len(freq_sim), n_log_vals)
    p, _ = curve_fit(fit_f, freq_sim[idcs], psd_sim[idcs])
    if fit_func == 'linear':
        params = pd.DataFrame({'Offset': p[0],
                               'Exponent': p[1]}, 
                               index=[0])
    elif fit_func == 'knee':

        #sometimes the knee frequency is unreasonably large (i.e. outside the fitting range)
        #take this as a marker that model fitting went wrong and return nans
        if 10**p[0] > freq.max():
            p = p * np.nan

        params = pd.DataFrame({'Knee Frequency Log10(Hz)': p[0],
                               'Knee Frequency (Hz)': 10**p[0],
                               'Offset': p[1],
                               'Exponent_1': p[2],
                               'Exponent_2': p[3],
                              }, index=[0])

    if np.isnan(p[0]):
        gof = pd.DataFrame({'mse': np.nan,
                            'r_squared': np.nan}, index=[0])
    else:
        #get goodness of fit
        residuals = np.log10(psd) - fit_f(np.log10(freq), *p)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((np.log10(psd)- np.mean(np.log10(psd))) ** 2)

        gof = pd.DataFrame({'mse': np.mean(residuals**2),
                            'r_squared': 1 - (ss_res / ss_tot)}, index=[0])

    return params, gof
