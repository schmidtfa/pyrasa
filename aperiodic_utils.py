import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import warnings


def fixed_model(x, b0, b):

    '''
    Specparams fixed fitting function.
    Use this to model aperiodic activity without a spectral knee
    '''

    y_hat = b0 - np.log10(x**b)

    return y_hat


def knee_model(x, b0, k, b1, b2):

    '''
    Model aperiodic activity with a spectral knee and a pre-knee slope.
    Use this to model aperiodic activity with a spectral knee
    '''

    y_hat = b0 - np.log10(x**b1 * (k + x**b2))

    return y_hat


def mixed_model(x, x1, b0, b1, b0_1, k, b2, b3):

    '''
    Fit the data using a piecewise function. 
    Where Part A is a fixed model fit and Part B is a fit that allows for a knee.
    Use this to model aperiodic activity that contains a spectral plateau.
    NOTE: This might require some additional testing action
    '''

    condlist = [x < x1,  x > x1]
    funclist = [lambda x: b0 - np.log10(x**b1), 
                lambda x: b0_1 - np.log10((x-x1)**b2 * (k + (x-x1)**b3))]
    y_hat = np.piecewise(x, condlist, funclist)
    return y_hat


def _get_gof(psd, psd_pred):

    ''' get goodness of fit (i.e. mean squared error and R2)'''
    
    residuals = np.log10(psd) - psd_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log10(psd) - np.mean(np.log10(psd))) ** 2)

    gof = pd.DataFrame({'mse': np.mean(residuals**2),
                        'r_squared': 1 - (ss_res / ss_tot)}, index=[0])
    return gof


def compute_slope(freq, psd, fit_func, fit_bounds=None):

    '''
    Compute the slope of the aperiodic spectrum. Enter a slope fitting function 
    and you get 2 pandas dataframes in return that contain the model parameters alongside the goodness of fit of the model.
        
    '''

    if freq[0] == 0:
        warnings.warn(f'The first frequency appears to be 0 this will result in slope fitting problems. \
                        Frequencies will be evaluated starting from the next highest, which is {freq[1]}Hz')
        freq = freq[1:]
        psd = psd[1:]

    curv_kwargs = {'maxfev': 10_000,
                   'ftol': 1e-5, 
                   'xtol': 1e-5, 
                   'gtol': 1e-5,}
    
    off_guess = [psd[0]] if fit_bounds == None else fit_bounds[0]
    exp_guess = [np.abs(np.log10(psd[0] / psd[-1]) / np.log10(freq[-1] / freq[0]))] if fit_bounds == None else fit_bounds[1]
    
    valid_slope_functions = ['fixed', 'knee', 'mixed']
    
    assert fit_func in valid_slope_functions, f'The slope fitting function has to be in {valid_slope_functions}'

    if fit_func == 'fixed':
        fit_f = fixed_model
        curv_kwargs['p0'] = np.array(off_guess + exp_guess)
        curv_kwargs['bounds'] = np.array([(0, 0), (np.inf, np.inf)]) 

        p, _ = curve_fit(fit_f, freq, np.log10(psd)) 

        
        params = pd.DataFrame({'Offset': p[0],
                               'Exponent': p[1],
                               'fit_type': 'fixed',
                                }, index=[0])
        psd_pred = fit_f(freq, *p)

    elif fit_func == 'knee':
        fit_f = knee_model
        #curve_fit_specs
        cumsum_psd = np.cumsum(psd)
        half_pw_freq = freq[np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()] 
        #make the knee guess the point where we have half the power in the spectrum seems plausible to me
        knee_guess = [half_pw_freq ** (exp_guess[0] + exp_guess[0])] 
        #convert knee freq to knee val which should be 2*exp_1 but this seems good enough
        curv_kwargs['p0'] = np.array(off_guess + knee_guess + exp_guess + exp_guess)
        print(curv_kwargs['p0'])
        curv_kwargs['bounds'] = ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf)) #make this optional
        #knee value should always be positive at least intuitively
        p, _ = curve_fit(fit_f, freq, np.log10(psd), **curv_kwargs)
        
        params = pd.DataFrame({'Offset': p[0],
                               'Knee': p[1],
                               'Exponent_1': p[2],
                               'Exponent_2': p[3],
                               'Knee Frequency (Hz)': p[1] ** (1. / (2*p[2] + p[3])),
                               'fit_type': 'knee',
                                }, index=[0])
        psd_pred = fit_f(freq, *p)

    elif fit_func == 'mixed':
        warnings.warn('Using the mixed model is super experimental and will probably fail')
        fit_f = mixed_model
        #curve_fit_specs
        cumsum_psd = np.cumsum(psd)
        half_pw_ix = np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()
        half_pw_freq = freq[half_pw_ix] 
        half_pw = [psd[half_pw_ix]]
        #make the knee guess the point where we have half the power in the spectrum seems plausible to me
        knee_guess = [half_pw_freq ** exp_guess[0]] #convert knee freq to knee val
        curv_kwargs['p0'] = np.array(knee_guess + off_guess + exp_guess + half_pw + knee_guess + exp_guess + exp_guess) #TODO: Really pay attention here
        curv_kwargs['bounds'] = ((0, 0, 0, 0, 0, 0, 0), 
                                 (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)) 
        #knee value should also always be positive at least intuitively
        p, _ = curve_fit(fit_f, freq, np.log10(psd), **curv_kwargs)
        
        params = pd.DataFrame({'switch_freq': p[0],
                               'Offset_1': p[1],
                               'Exponent_1': p[2],
                               'Offset_2': p[3],
                               'Knee': p[4],
                               'Knee Frequency (Hz)': p[4] ** (1. / (2*p[5] + p[6])),
                               'Exponent_3': p[5],
                               'Exponent_4': p[6],
                               'fit_type': 'expo + knee',
                                }, index=[0])
        psd_pred = fit_f(freq, *p)
    
    gof = _get_gof(psd, psd_pred)
    gof['fit_type'] = fit_func

    return params, gof



