"""Utilities for slope fitting."""

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pyrasa.utils.fit_funcs import AbstractFitFun, FixedFitFun, KneeFitFun
from pyrasa.utils.types import AperiodicFit


def _compute_aperiodic_model(
    aperiodic_spectrum: np.ndarray,
    freq: np.ndarray,
    fit_func: str | type[AbstractFitFun],
    scale_factor: float | int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """helper function to model the aperiodic spectrum"""

    if isinstance(fit_func, str):
        if fit_func == 'fixed':
            fit_func = FixedFitFun
        elif fit_func == 'knee':
            fit_func = KneeFitFun
        else:
            raise ValueError('fit_func should be either a string ("fixed", "knee") or of type AbastractFitFun')

    fit_f = fit_func(freq, aperiodic_spectrum, scale_factor=scale_factor)
    params, gof, pred = fit_f.fit_func()

    return params, gof, pred


def compute_aperiodic_model(
    aperiodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    fit_func: str | type[AbstractFitFun] = 'fixed',
    ch_names: Iterable | None = None,
    scale: bool = False,
    fit_bounds: tuple[float, float] | None = None,
) -> AperiodicFit:
    """
    Computes aperiodic parameters from the aperiodic spectrum using scipy's curve fitting function.

    This function can be used to model the aperiodic (1/f-like) component of the power spectrum. Per default, users can
    choose between a fixed or knee model fit or specify their own fit method see examples custom_fit_functions.ipynb
    for an example. The function returns the fitted parameters for each channel along with some
    goodness of fit metrics.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray
        A 1 or 2D array of power values for the aperiodic spectrum where the shape is
        expected to be either (Samples,) or (Channels, Samples).
    freqs : np.ndarray
        A 1D array of frequency values corresponding to the aperiodic spectrum.
    fit_func : str or type[AbstractFitFun], optional
        The fitting function to use. Can be "fixed" for a linear fit or "knee" for a fit that includes a
        knee (bend) in the spectrum or a class that is inherited from AbstractFitFun. The default is 'fixed'.
    ch_names : Iterable or None, optional
        Channel names corresponding to the aperiodic spectrum. If None, channels will be named numerically
        in ascending order. Default is None.
    scale : bool, optional
        Whether to scale the data to improve fitting accuracy. This is useful in cases where
        power values are very small (e.g., 1e-28), which may lead to numerical precision issues during fitting.
        After fitting, the parameters are rescaled to match the original data scale. Default is False.
    fit_bounds : tuple[float, float] or None, optional
        Tuple specifying the lower and upper frequency bounds for the fit function. If None, the entire frequency
        range is used. Otherwise, the spectrum is cropped to the specified bounds. Default is None.

    Returns
    -------
    AperiodicFit
        An object containing three pandas DataFrames:
            - aperiodic_params : pd.DataFrame
                A DataFrame containing the fitted aperiodic parameters for each channel.
            - gof : pd.DataFrame
                A DataFrame containing the goodness of fit metrics for each channel.
            - model : pd.DataFrame
                A DataFrame containing the predicted aperiodic model for each channel.

    Notes
    -----
    This function fits the aperiodic component of the power spectrum using scipy's curve fitting function.
    The fitting can be performed using either a simple linear model ('fixed') or a more complex model
    that includes a "knee" point, where the spectrum bends. The resulting parameters can help in
    understanding the underlying characteristics of the aperiodic component in the data.

    If the `fit_bounds` parameter is used, it ensures that only the specified frequency range is considered
    for fitting, which can be important to avoid fitting artifacts outside the region of interest.

    The `scale` parameter can be crucial when dealing with data that have extremely small values,
    as it helps to mitigate issues related to machine precision during the fitting process.

    The function asserts that the input data are of the correct type and shape, and raises warnings
    if the first frequency value is zero, as this can cause issues during model fitting.
    """

    assert isinstance(aperiodic_spectrum, np.ndarray), 'aperiodic_spectrum should be a numpy array.'
    if aperiodic_spectrum.ndim == 1:
        aperiodic_spectrum = aperiodic_spectrum[np.newaxis, :]
    assert aperiodic_spectrum.ndim == 2, 'Data shape needs to be either of shape (Channels, Samples) or (Samples, ).'  # noqa PLR2004

    assert isinstance(freqs, np.ndarray), 'freqs should be a numpy array.'
    assert freqs.ndim == 1, 'freqs needs to be of shape (freqs,).'

    assert isinstance(
        ch_names, list | tuple | np.ndarray | None
    ), 'Channel names should be of type list, tuple or numpy.ndarray or None'

    if fit_bounds is not None:
        fmin, fmax = freqs.min(), freqs.max()
        assert fit_bounds[0] > fmin, f'The selected lower bound is lower than the lowest frequency of {fmin}Hz'
        assert fit_bounds[1] < fmax, f'The selected upper bound is higher than the highest frequency of {fmax}Hz'
        freq_logical = np.logical_and(freqs >= fit_bounds[0], freqs <= fit_bounds[1])
        aperiodic_spectrum, freqs = aperiodic_spectrum[:, freq_logical], freqs[freq_logical]

    if freqs[0] == 0:
        warnings.warn(
            'The first frequency appears to be 0 this will result in aperiodic model fitting problems. '
            + 'Frequencies will be evaluated starting from the next highest in Hz'
        )
        freqs = freqs[1:]
        aperiodic_spectrum = aperiodic_spectrum[:, 1:]

    # generate channel names if not given
    if ch_names is None:
        ch_names = [str(i) for i in np.arange(aperiodic_spectrum.shape[0])]

    if scale:

        def num_zeros(decimal: int) -> float:
            return np.inf if decimal == 0 else -np.floor(np.log10(abs(decimal))) - 1

        scale_factor = 10 ** num_zeros(aperiodic_spectrum.min())
        aperiodic_spectrum = aperiodic_spectrum * scale_factor
    else:
        scale_factor = 1

    ap_list, gof_list, pred_list = [], [], []
    for ix, ch_name in enumerate(ch_names):
        params, gof, pred = _compute_aperiodic_model(
            aperiodic_spectrum=aperiodic_spectrum[ix],
            freq=freqs,
            fit_func=fit_func,
            scale_factor=scale_factor,
        )

        params['ch_name'] = ch_name
        gof['ch_name'] = ch_name
        pred['ch_name'] = ch_name

        ap_list.append(params)
        gof_list.append(gof)
        pred_list.append(pred)

    # combine & return
    return AperiodicFit(aperiodic_params=pd.concat(ap_list), gof=pd.concat(gof_list), model=pd.concat(pred_list))


def compute_aperiodic_model_sprint(
    aperiodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    fit_func: str | type[AbstractFitFun] = 'fixed',
    scale: bool = False,
    ch_names: Iterable | None = None,
    fit_bounds: tuple[float, float] | None = None,
) -> AperiodicFit:
    """
    Extracts aperiodic parameters from the aperiodic spectrogram using scipy's curve fitting
    function.

    This function computes aperiodic parameters for each time point in the spectrogram by applying either one of
    two different curve fitting functions (`fixed` or `knee`) or a custom function specified by user to the data.
    See examples custom_fit_functions.ipynb. The parameters, along with the goodness of
    fit for each time point, are returned in a concatenated format.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray
        A 2 or 3D array of power values from the aperiodic spectrogram, with shape (Frequencies, Time)
        or (Channels, Frequencies, Time).
    freqs : np.ndarray
        A 1D array of frequency values corresponding to the aperiodic spectrogram.
    times : np.ndarray
        A 1D array of time values corresponding to the aperiodic spectrogram.
    fit_func : str or type[AbstractFitFun], optional
        The fitting function to use. Can be "fixed" for a linear fit or "knee" for a fit that includes a
        knee (bend) in the spectrum or a class that is inherited from AbstractFitFun. The default is 'fixed'..
    ch_names : Iterable or None, optional
        Channel names corresponding to the aperiodic spectrogram. If None, channels will be named numerically
        in ascending order. Default is None.
    scale : bool, optional
        Whether to scale the data to improve fitting accuracy. This is useful when fitting a knee in cases where
        power values are very small, leading to numerical precision issues. Default is False.
    fit_bounds : tuple[float, float] or None, optional
        Tuple specifying the lower and upper frequency bounds for the fit function. If None, the entire frequency
        range is used. Otherwise, the spectrum is cropped to the specified bounds before fitting. Default is None.

    Returns
    -------
    AperiodicFit
        An object containing three pandas DataFrames:
            - aperiodic_params : pd.DataFrame
                A DataFrame containing the aperiodic parameters (e.g., Offset and Exponent)
                for each channel and each time point.
            - gof : pd.DataFrame
                A DataFrame containing the goodness of fit metrics for each channel and each time point.
            - model : pd.DataFrame
                A DataFrame containing the predicted aperiodic model for each channel and each time point.

    Notes
    -----
    This function iterates over each time point in the provided spectrogram to extract aperiodic parameters
    using the specified fit function. It leverages the `compute_aperiodic_model` function for individual fits
    at each time point, then combines the results across all time points into comprehensive DataFrames.

    The `fit_bounds` parameter allows for frequency range restrictions during fitting, which can help in focusing
    the analysis on a particular frequency band of interest.

    Scaling the data using the `scale` parameter can be particularly important when dealing with very small power
    values that might lead to poor fitting due to numerical precision limitations.

    """

    ap_t_list, gof_t_list, pred_t_list = [], [], []

    for ix, t in enumerate(times):
        aperiodic_fit = compute_aperiodic_model(
            aperiodic_spectrum[:, :, ix],
            freqs=freqs,
            fit_func=fit_func,
            ch_names=ch_names,
            fit_bounds=fit_bounds,
            scale=scale,
        )
        aperiodic_fit.aperiodic_params['time'] = t
        aperiodic_fit.gof['time'] = t
        aperiodic_fit.model['time'] = t

        ap_t_list.append(aperiodic_fit.aperiodic_params)
        gof_t_list.append(aperiodic_fit.gof)
        pred_t_list.append(aperiodic_fit.model)

    return AperiodicFit(aperiodic_params=pd.concat(ap_t_list), gof=pd.concat(gof_t_list), model=pd.concat(pred_t_list))
