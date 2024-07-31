"""Utilities for slope fitting."""

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def fixed_model(x: np.ndarray, b0: float, b: float) -> np.ndarray:
    """
    Specparams fixed fitting function.

    Use this to model aperiodic activity without a spectral knee.

    Parameters
    ----------
    x : np.ndarray
        Frequency values.
    b0 : float
        Offset parameter.
    b : float
        Exponent parameter.

    Returns
    -------
    np.ndarray
        Modeled aperiodic activity.
    """
    y_hat = b0 - np.log10(x**b)
    return y_hat


def knee_model(x: np.ndarray, b0: float, k: float, b1: float, b2: float) -> np.ndarray:
    """
    Model aperiodic activity with a spectral knee and a pre-knee slope.

    Use this to model aperiodic activity with a spectral knee.

    Parameters
    ----------
    x : np.ndarray
        Frequency values.
    b0 : float
        Offset parameter.
    k : float
        Knee parameter.
    b1 : float
        Pre-knee exponent parameter.
    b2 : float
        Post-knee exponent parameter.

    Returns
    -------
    np.ndarray
        Modeled aperiodic activity.
    """
    y_hat = b0 - np.log10(x**b1 * (k + x**b2))
    return y_hat


def _get_gof(psd: np.ndarray, psd_pred: np.ndarray, fit_func: str) -> pd.DataFrame:
    """
    Get goodness of fit (i.e. mean squared error and R2).

    Note:
    BIC and AIC currently assume OLS
    https://machinelearningmastery.com/probabilistic-model-selection-measures/.

    Parameters
    ----------
    psd : np.ndarray
        Power spectral density.
    psd_pred : np.ndarray
        Predicted power spectral density.
    fit_func : str
        Fitting function used ('fixed' or 'knee').

    Returns
    -------
    pd.DataFrame
        DataFrame containing goodness of fit metrics.
    """
    residuals = np.log10(psd) - psd_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((np.log10(psd) - np.mean(np.log10(psd))) ** 2)
    mse = np.mean(residuals**2)

    if fit_func == 'knee':
        k = 3  # k -> number of params
    elif fit_func == 'fixed':
        k = 1  # k -> number of params

    n = len(psd)
    bic = n * np.log(mse) + k * np.log(n)
    aic = n * np.log(mse) + 2 * k

    gof = pd.DataFrame({'mse': mse, 'r_squared': 1 - (ss_res / ss_tot), 'BIC': bic, 'AIC': aic}, index=[0])
    return gof


def _compute_slope(
    aperiodic_spectrum: np.ndarray,
    freq: np.ndarray,
    fit_func: str,
    fit_bounds: tuple | None = None,
    scale_factor: float | int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the slope of the aperiodic spectrum.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray
        Aperiodic power spectral density.
    freq : np.ndarray
        Frequency values.
    fit_func : str
        Fitting function to use ('fixed' or 'knee').
    fit_bounds : tuple or None, optional
        Bounds for the fitting function. Defaults to None.
    scale_factor : float or int, optional
        Scaling factor for the data. Defaults to 1.

    Returns
    -------
    tuple of pd.DataFrame
        DataFrames containing the aperiodic parameters and goodness of fit metrics.
    """
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

    valid_slope_functions = ['fixed', 'knee']
    assert fit_func in valid_slope_functions, f'The slope fitting function has to be in {valid_slope_functions}'

    if fit_func == 'fixed':
        fit_f = fixed_model

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
        fit_f = knee_model  # type: ignore
        # curve_fit_specs
        cumsum_psd = np.cumsum(aperiodic_spectrum)
        half_pw_freq = freq[np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()]
        # make the knee guess the point where we have half the power in the spectrum seems plausible to me
        knee_guess = [half_pw_freq ** (exp_guess[0] + exp_guess[0])]
        # convert knee freq to knee val which should be 2*exp_1 but this seems good enough
        curv_kwargs['p0'] = np.array(off_guess + knee_guess + exp_guess + exp_guess)  # type: ignore
        # print(curv_kwargs['p0'])
        # make this optional
        curv_kwargs['bounds'] = ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf))  # type: ignore
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


def compute_slope(
    aperiodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    fit_func: str,
    ch_names: Iterable = (),
    scale: bool = False,
    fit_bounds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.

    The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
    as well as the respective goodness of fit.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray
        Power values for the aperiodic spectrum extracted using IRASA (shape: channel x frequency).
    freqs : np.ndarray
        Frequency values for the aperiodic spectrum.
    fit_func : str
        Fitting function to use ('fixed' or 'knee').
    ch_names : Iterable, optional
        Channel names ordered according to the periodic spectrum. Defaults to an empty iterable.
    scale : bool, optional
        Scale the data by a factor to improve fitting. Defaults to False.
    fit_bounds : tuple of float or None, optional
        Lower and upper bounds for the fit function. Defaults to None.

    Returns
    -------
    tuple of pd.DataFrame
        DataFrames containing the aperiodic parameters and goodness of fit metrics.

    Raises
    ------
    AssertionError
        If the input parameters are invalid.
    """
    assert isinstance(aperiodic_spectrum, np.ndarray), 'aperiodic_spectrum should be a numpy array.'
    if aperiodic_spectrum.ndim == 1:
        aperiodic_spectrum = aperiodic_spectrum[np.newaxis, :]
    assert aperiodic_spectrum.ndim == 2, 'Data shape needs to be either of shape (Channels, Samples) or (Samples, ).'  # noqa PLR2004

    assert isinstance(freqs, np.ndarray), 'freqs should be a numpy array.'
    assert freqs.ndim == 1, 'freqs needs to be of shape (freqs,).'

    assert isinstance(
        ch_names, list | tuple | np.ndarray
    ), 'Channel names should be of type list, tuple or numpy.ndarray'

    if fit_bounds is not None:
        fmin, fmax = freqs.min(), freqs.max()
        assert fit_bounds[0] > fmin, f'The selected lower bound is lower than the lowest frequency of {fmin}Hz'
        assert fit_bounds[1] < fmax, f'The selected upper bound is higher than the highest frequency of {fmax}Hz'

    if freqs[0] == 0:
        warnings.warn(
            'The first frequency appears to be 0 this will result in slope fitting problems. '
            + 'Frequencies will be evaluated starting from the next highest in Hz'
        )
        freqs = freqs[1:]
        aperiodic_spectrum = aperiodic_spectrum[:, 1:]

    # generate channel names if not given
    if len(ch_names) == 0:
        ch_names = np.arange(aperiodic_spectrum.shape[0])

    if scale:

        def num_zeros(decimal: int) -> float:
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


def compute_slope_sprint(
    aperiodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    fit_func: str,
    ch_names: Iterable = (),
    fit_bounds: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract aperiodic parameters from the aperiodic spectrogram extracted from IRASA.

    The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
    as well as the respective goodness of fit.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray
        Power values for the aperiodic spectrogram extracted using IRASA (shape: channel x frequency).
    freqs : np.ndarray
        Frequency values for the aperiodic spectrogram.
    times : np.ndarray
        Time values for the aperiodic spectrogram.
    fit_func : str
        Fitting function to use ('fixed' or 'knee').
    ch_names : Iterable, optional
        Channel names ordered according to the periodic spectrum. Defaults to an empty iterable.
    fit_bounds : tuple of float or None, optional
        Lower and upper bounds for the fit function. Defaults to None.

    Returns
    -------
    tuple of pd.DataFrame
        DataFrames containing the aperiodic parameters and goodness of fit metrics.

    Raises
    ------
    AssertionError
        If the input parameters are invalid.
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
