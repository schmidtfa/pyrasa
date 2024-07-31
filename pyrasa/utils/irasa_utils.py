"""Utilities for signal decompositon using IRASA."""

from collections.abc import Callable
from copy import copy

import numpy as np
import scipy.signal as dsp
from scipy.signal import ShortTimeFFT


def _crop_data(
    band: list | tuple, freqs: np.ndarray, psd_aperiodic: np.ndarray, psd_periodic: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop spectra to a defined frequency range.

    Parameters
    ----------
    band : list or tuple
        Frequency range to crop to.
    freqs : np.ndarray
        Frequency values.
    psd_aperiodic : np.ndarray
        Aperiodic power spectral density.
    psd_periodic : np.ndarray
        Periodic power spectral density.
    axis : int
        Axis along which to crop.

    Returns
    -------
    tuple of np.ndarray
        Cropped frequency values, aperiodic PSD, and periodic PSD.
    """
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=axis)
    psd_periodic = np.compress(~mask_freqs, psd_periodic, axis=axis)

    return freqs, psd_aperiodic, psd_periodic


def _gen_time_from_sft(SFT: type[dsp.ShortTimeFFT], sgramm: np.ndarray) -> np.ndarray:  # noqa N803
    """
    Generate time from SFT object.

    Parameters
    ----------
    SFT : type[dsp.ShortTimeFFT]
        Short-time Fourier transform object.
    sgramm : np.ndarray
        Spectrogram data.

    Returns
    -------
    np.ndarray
        Time values.
    """
    tmin, tmax = SFT.extent(sgramm.shape[-1])[:2]
    delta_t = SFT.delta_t

    time = np.arange(tmin, tmax, delta_t)
    return time


def _find_nearest(sgramm_ud: np.ndarray, time_array: np.ndarray, time_value: float) -> np.ndarray:
    """
    Find the nearest time point in an up/downsampled spectrogram.

    Parameters
    ----------
    sgramm_ud : np.ndarray
        Up/downsampled spectrogram.
    time_array : np.ndarray
        Array of time values.
    time_value : float
        Time value to find the nearest point to.

    Returns
    -------
    np.ndarray
        Spectrogram at the nearest time point.
    """
    idx = (np.abs(time_array - time_value)).argmin()

    if idx < sgramm_ud.shape[2]:
        sgramm_sel = sgramm_ud[:, :, idx]

    elif idx == sgramm_ud.shape[2]:
        sgramm_sel = sgramm_ud[:, :, idx - 1]

    return sgramm_sel


def _get_windows(
    nperseg: int, dpss_settings: dict, win_func: Callable, win_func_kwargs: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a window function used for tapering.

    Parameters
    ----------
    nperseg : int
        Length of each segment.
    dpss_settings : dict
        Settings for DPSS window.
    win_func : Callable
        Window function.
    win_func_kwargs : dict
        Additional arguments for the window function.

    Returns
    -------
    tuple of np.ndarray
        Window function and ratios.
    """
    low_bias_ratio = 0.9
    min_time_bandwidth = 2.0
    win_func_kwargs = copy(win_func_kwargs)

    # special settings in case multitapering is required
    if win_func == dsp.windows.dpss:
        time_bandwidth = dpss_settings['time_bandwidth']
        if time_bandwidth > min_time_bandwidth:
            raise ValueError(f'time_bandwidth should be >= {min_time_bandwidth} for good tapers')

        n_taps = int(np.floor(time_bandwidth - 1))
        win_func_kwargs.update(
            {
                'NW': time_bandwidth / 2,  # half width
                'Kmax': n_taps,
                'sym': False,
                'return_ratios': True,
            }
        )
        win, ratios = win_func(nperseg, **win_func_kwargs)
        if dpss_settings['low_bias']:
            win = win[ratios > low_bias_ratio]
            ratios = ratios[ratios > low_bias_ratio]
    else:
        win = [win_func(nperseg, **win_func_kwargs)]
        ratios = None

    return win, ratios


def _check_irasa_settings(irasa_params: dict, hset_info: tuple) -> None:
    """
    Check if the input parameters for IRASA are specified correctly.

    Parameters
    ----------
    irasa_params : dict
        Parameters for IRASA.
    hset_info : tuple
        Information about the hset.

    Raises
    ------
    AssertionError
        If any of the input parameters are invalid.
    """
    valid_hset_shape = 3
    assert isinstance(irasa_params['data'], np.ndarray), 'Data should be a numpy array.'

    # check if hset is specified correctly
    assert isinstance(
        hset_info, tuple | list | np.ndarray
    ), 'hset should be a tuple, list or numpy array of (min, max, step)'
    assert np.shape(hset_info)[0] == valid_hset_shape, 'shape of hset_info should be 3 i.e. (min, max, step)'

    # check that evaluated range fits with the data settings
    nyquist = irasa_params['fs'] / 2
    hmax = np.max(hset_info)
    band_evaluated: tuple[float, float] = (irasa_params['band'][0] / hmax, irasa_params['band'][1] * hmax)
    assert band_evaluated[0] > 0, 'The evaluated frequency range is 0 or lower this makes no sense'
    assert band_evaluated[1] < nyquist, (
        f'The evaluated frequency range goes up to {np.round(band_evaluated[1], 2)}Hz '
        'which is higher than Nyquist (fs / 2)'
    )

    filter_settings: list[float] = list(irasa_params['filter_settings'])
    if filter_settings[0] is None:
        filter_settings[0] = band_evaluated[0]
    if filter_settings[1] is None:
        filter_settings[1] = band_evaluated[1]

    assert np.logical_and(band_evaluated[0] >= filter_settings[0], band_evaluated[1] <= filter_settings[1]), (
        f'You run IRASA in a frequency range from'
        f'{np.round(band_evaluated[0], irasa_params["hset_accuracy"])} -'
        f'{np.round(band_evaluated[1], irasa_params["hset_accuracy"])}Hz. \n'
        'Your settings specified in "filter_settings" indicate that you have '
        'a bandpass filter from '
        f'{np.round(filter_settings[0], irasa_params["hset_accuracy"])} - '
        f'{np.round(filter_settings[1], irasa_params["hset_accuracy"])}Hz. \n'
        'This means that your evaluated range likely contains filter artifacts. \n'
        'Either change your filter settings, adjust hset or the parameter "band" accordingly. \n'
        f'You want to make sure that band[0] / hset.max() '
        f'> {np.round(filter_settings[0], irasa_params["hset_accuracy"])} '
        f'and that band[1] * hset.max() < {np.round(filter_settings[1], irasa_params["hset_accuracy"])}'
    )


# Calculate original spectrum
def _compute_psd_welch(
    data: np.ndarray,
    fs: int,
    nperseg: int | None,
    win_kwargs: dict,
    dpss_settings: dict,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str = 'constant',
    return_onesided: bool = True,
    scaling: str = 'density',
    axis: int = -1,
    average: str = 'mean',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral densities using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    fs : int
        Sampling frequency.
    nperseg : int or None
        Length of each segment.
    win_kwargs : dict
        Additional arguments for the window function.
    dpss_settings : dict
        Settings for DPSS window.
    noverlap : int or None, optional
        Number of points to overlap between segments. Defaults to None.
    nfft : int or None, optional
        Length of the FFT used. Defaults to None.
    detrend : str, optional
        Specifies how to detrend each segment. Defaults to 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. Defaults to True.
    scaling : str, optional
        Selects between computing the power spectral density ('density') and
        the power spectrum ('spectrum'). Defaults to 'density'.
    axis : int, optional
        Axis along which the periodogram is computed. Defaults to -1.
    average : str, optional
        Method to use when averaging periodograms. Defaults to 'mean'.

    Returns
    -------
    tuple of np.ndarray
        Frequencies and power spectral densities.
    """
    if nperseg is None:
        nperseg = data.shape[-1]
    win, ratios = _get_windows(nperseg, dpss_settings, **win_kwargs)

    psds = []
    for cur_win in win:
        freq, psd = dsp.welch(
            data,
            fs=fs,
            window=cur_win,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            average=average,
        )
        psds.append(psd)

    if ratios is None:
        psd = np.mean(psds, axis=0)
    else:
        weighted_psds = [ratios[ix] * cur_sgramm for ix, cur_sgramm in enumerate(psds)]
        psd = np.sum(weighted_psds, axis=0) / np.sum(ratios)

    return freq, psd


def _compute_sgramm(  # noqa C901
    x: np.ndarray,
    fs: int,
    mfft: int,
    hop: int,
    win_duration: float,
    dpss_settings: dict,
    win_kwargs: dict,
    up_down: str | None = None,
    h: int | None = None,
    time_orig: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrograms.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : int
        Sampling frequency.
    mfft : int
        Number of FFT points.
    hop : int
        Hop size.
    win_duration : float
        Window duration in seconds.
    dpss_settings : dict
        Settings for DPSS window.
    win_kwargs : dict
        Additional arguments for the window function.
    up_down : str or None, optional
        Upsampling or downsampling direction. Defaults to None.
    h : int or None, optional
        Upsampling or downsampling factor. Defaults to None.
    time_orig : np.ndarray or None, optional
        Original time values. Defaults to None.

    Returns
    -------
    tuple of np.ndarray
        Frequencies, time values, and spectrogram.
    """
    if h is None:
        nperseg = int(np.floor(fs * win_duration))
    elif np.logical_and(h is not None, up_down == 'up'):
        nperseg = int(np.floor(fs * win_duration * h))
        hop = int(hop * h)
    elif np.logical_and(h is not None, up_down == 'down'):
        nperseg = int(np.floor(fs * win_duration / h))
        hop = int(hop / h)

    win, ratios = _get_windows(nperseg, dpss_settings, **win_kwargs)

    sgramms = []
    for cur_win in win:
        SFT = ShortTimeFFT(cur_win, hop=hop, mfft=mfft, fs=fs, scale_to='psd')  # noqa N806
        cur_sgramm = SFT.spectrogram(x, detr='constant')
        sgramms.append(cur_sgramm)

    if ratios is None:
        sgramm = np.mean(sgramms, axis=0)
    else:
        weighted_sgramms = [ratios[ix] * cur_sgramm for ix, cur_sgramm in enumerate(sgramms)]
        sgramm = np.sum(weighted_sgramms, axis=0) / np.sum(ratios)

    time = _gen_time_from_sft(SFT, x)
    freq = SFT.f[SFT.f > 0]

    # subsample the upsampled data in the time domain to allow averaging
    # This is necessary as division by h can cause slight rounding differences that
    # result in actual unintended temporal differences in up/dw for very long segments.
    if time_orig is not None:
        sgramm = np.array([_find_nearest(sgramm, time, t) for t in time_orig])
        max_t_ix = time_orig.shape[0]
        # swapping axes is necessitated by _find_nearest
        sgramm = np.swapaxes(
            np.swapaxes(sgramm[:max_t_ix, :, :], 1, 2), 0, 2
        )  # cut time axis for up/downsampled data to allow averaging

    sgramm = np.squeeze(sgramm)  # bring in proper format

    return freq, time, sgramm
