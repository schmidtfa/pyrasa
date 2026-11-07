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


def compute_aperiodic_model(  # noqa: C901
    aperiodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    fit_func: str | type[AbstractFitFun] = 'fixed',
    ch_names: Iterable | None = None,
    scale: bool = False,
    scale_percentile: float = 5.0,
    fit_bounds: tuple[float, float] | None = None,
) -> AperiodicFit:
    """
    Fit a parametric model to the aperiodic component of a power spectrum.

    Fit the aperiodic (1/f-like) component of the power spectrum separately
    for each channel. Two models are available by default: a fixed model
    describing a power-law relationship without a knee, and a knee model
    that additionally captures a bend in the aperiodic spectrum.

    Custom fitting functions can be provided as subclasses of
    `AbstractFitFun`.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray, shape (n_freqs,) or (n_channels, n_freqs)
        Aperiodic power spectral values for one or multiple channels.
        One-dimensional input is treated as a single-channel spectrum.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the aperiodic spectrum, in Hz.
    fit_func : str | type[AbstractFitFun], default 'fixed'
        Model used to fit the aperiodic spectrum. Options are:

        - 'fixed': Fit a power-law model without a knee, characterized
        by an offset and exponent.
        - 'knee': Fit a model including a spectral knee.
        - A subclass of `AbstractFitFun` defining a custom fitting model.

    ch_names : list | tuple | np.ndarray | None, optional
        Channel names corresponding to the aperiodic spectrum.
        If None, channels are assigned numerical string labels
        starting from '0'.
    scale : bool, default False
        Whether to scale the aperiodic spectrum before fitting to improve
        numerical stability. This can be useful when power values are
        very small. The scaling factor is passed to the fitting model
        to account for the original data scale.
    scale_percentile : float, default 5.0
        Percentile of the aperiodic power values used to determine the
        scaling factor. Must be between 0 and 100. The percentile is
        calculated across all channels and frequencies included in the fit,
        providing a common scaling factor. A value of 0 uses the minimum
        power value. Only used when `scale=True`.
    fit_bounds : tuple[float, float] | None, optional
        Lower and upper frequency bounds for model fitting, in Hz.
        Both bounds must lie strictly within the available frequency
        range. Frequencies equal to the specified bounds are included.
        If None, the full available frequency range is used.

    Returns
    -------
    AperiodicFit
        Object containing three pandas DataFrames:

        - `aperiodic_params`: Fitted model parameters for each channel,
        including a `ch_name` column.
        - `gof`: Goodness-of-fit metrics for each channel, including
        a `ch_name` column.
        - `model`: Predicted aperiodic power values for each channel
        at the fitted frequencies, including a `ch_name` column.

    Notes
    -----
    If the first frequency value is 0 Hz, a warning is issued and this
    frequency is excluded from model fitting.

    When `scale=True`, the scaling factor is determined after applying
    the specified frequency bounds and excluding 0 Hz, if present.

    For the built-in fixed and knee models, goodness-of-fit metrics are
    calculated in log10 power space, whereas the returned model predictions
    are expressed in the original linear power scale.

    See Also
    --------
    compute_aperiodic_model_sprint
    pyrasa.utils.fit_funcs.AbstractFitFun
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

    if not 0 <= scale_percentile <= 100:  # noqa: PLR2004
        raise ValueError('scale_percentile must be between 0 and 100.')

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

        reference_power = np.percentile(aperiodic_spectrum, scale_percentile)
        scale_factor = 10.0 ** num_zeros(reference_power)

        if not np.isfinite(scale_factor):
            raise ValueError('Scaling factor is not finite.')

        scaled_spectrum = aperiodic_spectrum * scale_factor

        if not np.all(np.isfinite(scaled_spectrum)):
            raise ValueError('Scaling produced non-finite values.')

        aperiodic_spectrum = scaled_spectrum
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
    Fit parametric aperiodic models to a time-resolved power spectrum.

    Fit the aperiodic (1/f-like) component of a spectrogram separately
    for each channel and time point.

    Two models are available by default: a fixed model describing a
    power-law relationship without a knee, and a knee model that
    additionally captures a bend in the aperiodic spectrum. Custom
    fitting functions can be provided as subclasses of `AbstractFitFun`.

    The fitted parameters, goodness-of-fit metrics, and model predictions
    are combined across time points and annotated with their
    corresponding time values.

    Parameters
    ----------
    aperiodic_spectrum : np.ndarray, shape (n_channels, n_freqs, n_times)
        Time-resolved aperiodic power spectral values. The last
        dimension represents time.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the aperiodic spectrogram,
        in Hz.
    times : np.ndarray, shape (n_times,)
        Time values corresponding to the aperiodic spectrogram,
        in seconds.
    fit_func : str | type[AbstractFitFun], default 'fixed'
        Model used to fit the aperiodic spectrum. Options are:

        - 'fixed': Fit a power-law model without a knee, characterized
        by an offset and exponent.
        - 'knee': Fit a model including a spectral knee.
        - A subclass of `AbstractFitFun` defining a custom fitting model.

    scale : bool, default False
        Whether to scale the aperiodic spectrum before fitting to improve
        numerical stability. This can be useful when power values are
        very small. The scaling factor is passed to the fitting model
        to account for the original data scale.
    ch_names : list | tuple | np.ndarray | None, optional
        Channel names corresponding to the aperiodic spectrogram.
        If None, channels are assigned numerical string labels
        starting from '0'.
    fit_bounds : tuple[float, float] | None, optional
        Lower and upper frequency bounds for model fitting, in Hz.
        Both bounds must lie strictly within the available frequency
        range. Frequencies equal to the specified bounds are included.
        If None, the full available frequency range is used.

    Returns
    -------
    AperiodicFit
        Object containing three pandas DataFrames:

        - `aperiodic_params`: Fitted model parameters for each channel
        and time point, including `ch_name` and `time` columns.
        - `gof`: Goodness-of-fit metrics for each channel and time point,
        including `ch_name` and `time` columns.
        - `model`: Predicted aperiodic power values for each channel
        and time point at the fitted frequencies, including
        `ch_name` and `time` columns.

    Notes
    -----
    Models are fitted independently for each channel and time point.
    No averaging across time points is performed.

    Fitting is performed using `compute_aperiodic_model`. When scaling
    is enabled, a separate scaling factor is determined for each time
    point using the default 5th percentile across its channels and
    included frequencies.

    For the built-in fixed and knee models, goodness-of-fit metrics are
    calculated in log10 power space, whereas the returned model predictions
    are expressed in the original linear power scale.

    See Also
    --------
    compute_aperiodic_model
    pyrasa.utils.fit_funcs.AbstractFitFun
    """

    ap_t_list, gof_t_list, pred_t_list = [], [], []

    is_2d = 2
    if aperiodic_spectrum.ndim == is_2d:
        aperiodic_spectrum = aperiodic_spectrum[np.newaxis, :, :]

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
