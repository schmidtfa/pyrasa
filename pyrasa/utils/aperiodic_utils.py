"""Utilities for slope fitting."""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fixed_model(x, b0, b):
    """
    Specparams fixed fitting function.
    Use this to model aperiodic activity without a spectral knee
    """

    y_hat = b0 - np.log10(x**b)

    return y_hat


def knee_model(x, b0, k, b1, b2):
    """
    Model aperiodic activity with a spectral knee and a pre-knee slope.
    Use this to model aperiodic activity with a spectral knee
    """

    y_hat = b0 - np.log10(x**b1 * (k + x**b2))

    return y_hat


def _get_gof(psd, psd_pred, fit_func):
    """
    get goodness of fit (i.e. mean squared error and R2)
    BIC and AIC currently assume OLS
    https://machinelearningmastery.com/probabilistic-model-selection-measures/
    """

    residuals = np.log10(psd) - psd_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log10(psd) - np.mean(np.log10(psd))) ** 2)
    mse = np.mean(residuals**2)

    if fit_func == 'knee':
        k = 3  # k -> number of params
    elif fit_func == 'fixed':
        k = 1

    n = len(psd)
    bic = n * np.log(mse) + k * np.log(n)
    aic = n * np.log(mse) + 2 * k

    gof = pd.DataFrame({'mse': mse, 'r_squared': 1 - (ss_res / ss_tot), 'BIC': bic, 'AIC': aic}, index=[0])
    return gof


def _compute_slope(aperiodic_spectrum, freq, fit_func, fit_bounds=None, scale_factor=1):
    """get the slope of the aperiodic spectrum"""

    curv_kwargs = {
        'maxfev': 10_000,
        'ftol': 1e-5,
        'xtol': 1e-5,
        'gtol': 1e-5,
    }

    off_guess = [aperiodic_spectrum[0]] if fit_bounds is None else fit_bounds[0]
    exp_guess = (
        [np.abs(np.log10(aperiodic_spectrum[0] / aperiodic_spectrum[-1]) / np.log10(freq[-1] / freq[0]))]
        if fit_bounds is None
        else fit_bounds[1]
    )

    valid_slope_functions = ['fixed', 'knee', 'mixed']
    assert fit_func in valid_slope_functions, f'The slope fitting function has to be in {valid_slope_functions}'

    if fit_func == 'fixed':
        fit_f = fixed_model
        curv_kwargs['p0'] = np.array(off_guess + exp_guess)
        curv_kwargs['bounds'] = np.array([(0, 0), (np.inf, np.inf)])

        p, _ = curve_fit(fit_f, freq, np.log10(aperiodic_spectrum))

        params = pd.DataFrame(
            {
                'Offset': p[0],
                'Exponent': p[1],
                'fit_type': 'fixed',
            },
            index=[0],
        )
        psd_pred = fit_f(freq, *p)

    elif fit_func == 'knee':
        fit_f = knee_model
        # curve_fit_specs
        cumsum_psd = np.cumsum(aperiodic_spectrum)
        half_pw_freq = freq[np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()]
        # make the knee guess the point where we have half the power in the spectrum seems plausible to me
        knee_guess = [half_pw_freq ** (exp_guess[0] + exp_guess[0])]
        # convert knee freq to knee val which should be 2*exp_1 but this seems good enough
        curv_kwargs['p0'] = np.array(off_guess + knee_guess + exp_guess + exp_guess)
        # print(curv_kwargs['p0'])
        curv_kwargs['bounds'] = ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))  # make this optional
        # knee value should always be positive at least intuitively
        p, _ = curve_fit(fit_f, freq, np.log10(aperiodic_spectrum), **curv_kwargs)

        params = pd.DataFrame(
            {
                'Offset': p[0] / scale_factor,
                'Knee': p[1],
                'Exponent_1': p[2],
                'Exponent_2': p[3],
                'Knee Frequency (Hz)': p[1] ** (1.0 / (2 * p[2] + p[3])),
                'fit_type': 'knee',
            },
            index=[0],
        )
        psd_pred = fit_f(freq, *p)

    gof = _get_gof(aperiodic_spectrum, psd_pred, fit_func)
    gof['fit_type'] = fit_func

    return params, gof


def compute_slope(aperiodic_spectrum, freqs, fit_func, ch_names=[], scale=True, fit_bounds=None):
    """
    This function can be used to extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.
    The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
    as well as the respective goodness of fit.

    Parameters: aperiodic_spectrum : 2d array
                    Power values for the aeriodic spectrum extracted using IRASA shape (channel x frequency)
                freqs : 1d array
                    Frequency values for the aperiodic spectrum
                fit_func : string
                    Can be either "fixed" or "knee".
                ch_names : list, optional, default: []
                    Channel names ordered according to the periodic spectrum.
                    If empty channel names are given as numbers in ascending order.
                scale : bool
                    scale the data by a factor of x to improve fitting.
                    This is helpful when fitting a knee and power values are very small eg. 1e-28,
                    in which case curve fits struggles to find the proper MSE (seems to be a machine precision issue).
                    Finally the data are rescaled to return the offset in the magnitude of the original data.
                fit_bounds : None, tuple
                    Lower and upper bound for the fit function, should be None if the whole frequency range is desired.
                    Otherwise a tuple of (lower, upper)

    Returns:    df_aps: DataFrame
                    DataFrame containing the center frequency, bandwidth and peak height for each channel
                df_gof: DataFrame
                    DataFrame containing the goodness of fit of the specific fit function for each channel.

    """

    assert isinstance(aperiodic_spectrum, np.ndarray), 'aperiodic_spectrum should be a numpy array.'
    assert aperiodic_spectrum.ndim == 2, 'aperiodic_spectrum needs to be of shape (Channels, Samples).'  # noqa PLR2004

    assert isinstance(freqs, np.ndarray), 'freqs should be a numpy array.'
    assert freqs.ndim == 1, 'freqs needs to be of shape (freqs,).'

    assert isinstance(ch_names, list), 'Channel names should be of type list'

    if fit_bounds is not None:
        fmin, fmax = freqs.min(), freqs.max()
        assert fit_bounds[0] > fmin, f'The selected lower bound is lower than the lowest frequency of {fmin}Hz'
        assert fit_bounds[0] < fmax, f'The selected upper bound is higher than the highest frequency of {fmax}Hz'

    if freqs[0] == 0:
        warnings.warn(f'The first frequency appears to be 0 this will result in slope fitting problems. \
                        Frequencies will be evaluated starting from the next highest, which is {freqs[1]}Hz')
        freqs = freqs[1:]
        aperiodic_spectrum = aperiodic_spectrum[1:]

    # generate channel names if not given
    if len(ch_names) == 0:
        ch_names = np.arange(aperiodic_spectrum.shape[0])

    if scale:

        def num_zeros(decimal):
            return np.inf if decimal == 0 else -np.floor(np.log10(abs(decimal))) - 1

        scale_factor = 10 ** num_zeros(aperiodic_spectrum.min())
        aperiodic_spectrum = aperiodic_spectrum * scale_factor
    else:
        scale_factor = 1

    ap_list, gof_list = [], []
    for ix, ch_name in enumerate(ch_names):
        params, gof = _compute_slope(
            aperiodic_spectrum=aperiodic_spectrum[ix],
            freq=freqs,
            fit_func=fit_func,
            scale_factor=scale_factor,
            fit_bounds=fit_bounds,
        )

        params['ch_name'] = ch_name
        gof['ch_name'] = ch_name

        ap_list.append(params)
        gof_list.append(gof)

    # combine & return
    df_aps = pd.concat(ap_list)
    df_gof = pd.concat(gof_list)

    return df_aps, df_gof


def compute_slope_sprint(aperiodic_spectrum, freqs, times, fit_func, ch_names=[], fit_bounds=None):
    """
    This function can be used to extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.
    The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
    as well as the respective goodness of fit.

    Parameters: aperiodic_spectrum : 2d array
                    Power values for the aeriodic spectrum extracted using IRASA shape (channel x frequency)
                freqs : 1d array
                    Frequency values for the aperiodic spectrum
                fit_func : string
                    Can be either "fixed" or "knee".
                ch_names : list, optional, default: []
                    Channel names ordered according to the periodic spectrum.
                    If empty channel names are given as numbers in ascending order.
                fit_bounds : None, tuple
                    Lower and upper bound for the fit function, should be None if the whole frequency range is desired.
                    Otherwise a tuple of (lower, upper)

    Returns:    df_aps: DataFrame
                    DataFrame containing the center frequency, bandwidth and peak height for each channel
                df_gof: DataFrame
                    DataFrame containing the goodness of fit of the specific fit function for each channel.

    """

    ap_t_list, gof_t_list = [], []

    for ix, t in enumerate(times):
        cur_aps, cur_gof = compute_slope(
            aperiodic_spectrum[:, :, ix], freqs=freqs, fit_func=fit_func, ch_names=ch_names, fit_bounds=fit_bounds
        )
        cur_aps['time'] = t
        cur_gof['time'] = t

        ap_t_list.append(cur_aps)
        gof_t_list.append(cur_gof)

    df_ap_time = pd.concat(ap_t_list)
    df_gof_time = pd.concat(gof_t_list)

    return df_ap_time, df_gof_time
