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


def _gen_equally_distant_log_vals(freq, psd, interp_factor=1000, n_log_vals=100):

    '''
    
    This helper function selects equally spaced values in log/log 
    -> This is needed for line fits in loglog to not nias the estimate by the higher values
    
    '''
    #need to interpolate as less points in the beginning compared to the end in loglog
    f_interp = interp1d(np.log10(freq),np.log10(psd), kind='linear')
    freq_sim = np.log10(np.linspace(freq.min(), freq.max(), num=len(freq)*interp_factor))
    psd_sim = f_interp(freq_sim)
    idcs = gen_log_space(len(freq_sim), n_log_vals) #we need equally distant points in log-log

    return freq_sim[idcs], psd_sim[idcs]


def lin_reg(x, a, b):

    '''
    This is legacy stuff!!
    Linear fit w/o knee 
    NOTE: Data should be in loglog coordinates before running this
    '''

    y_hat = a + b * x

    return y_hat


def piecewise_linear(x, x1, b, k1, k2):

    '''
    This is legacy stuff!!
    Knee Fit using a piecewise linear regression 
    NOTE: Data should be in loglog coordinates before running this
    '''

    condlist = [x < x1,  x > x1]
    funclist = [lambda x: k1*x + b, lambda x: k1*x + b + k2*(x-x1)]
    return np.piecewise(x, condlist, funclist)



def expo_reg(x, a, k, b):

    '''Uses specparams fitting function with a knee
    NOTE: This should only be used for educative purposes, 
    if the slope pre-knee != 0 this will lead to fucked up results.'''

    y_hat = a - np.log10(k + x**b)

    return y_hat



    elif fit_func == 'double_exponential':
            
    fit_f = double_expo_reg
    p, _ = curve_fit(fit_f, freq, np.log10(psd)) 
    
    params = pd.DataFrame({'Knee': p[2],
                            #'Knee Frequency (Hz)': p[2] ** (1. / p[3]), 
                            #TODO: In light of 2 exponents the knee frequency likely needs to be calculated differently
                            'Offset': p[0],
                            'Exponent_1': p[1],
                            'Exponent_2': p[3],
                            }, index=[0])
    
    psd_pred = fit_f(freq, *p)
    gof = _get_gof(psd, psd_pred)

elif fit_func == 'linear_loglog':
    fit_f = lin_reg
    freq_log, psd_log = _gen_equally_distant_log_vals(freq, psd)
    p, _ = curve_fit(fit_f, freq_log, psd_log)
    params = pd.DataFrame({'Offset': p[0],
                            'Exponent': p[1]}, 
                            index=[0])
    psd_pred = fit_f(np.log10(freq), *p) #need freq as log10 input here
    gof = _get_gof(psd, psd_pred)

elif fit_func == 'piecewise_linear_loglog':
    fit_f = piecewise_linear
    freq_log, psd_log = _gen_equally_distant_log_vals(freq, psd)
    p, _ = curve_fit(fit_f, freq_log, psd_log)
    #sometimes the knee frequency is unreasonably large (i.e. outside the fitting range)
    #take this as a marker that model fitting went wrong and return nans
    if 10**p[0] > freq.max():
        p = p * np.nan

    params = pd.DataFrame({'Knee Frequency (Hz)': 10**p[0], #lets unlog it
                            'Offset': p[1],
                            'Exponent_1': p[2],
                            'Exponent_2': p[3],
                            }, index=[0])
    psd_pred = fit_f(np.log10(freq), *p) #need freq as log10 input here
    gof = _get_gof(psd, psd_pred)