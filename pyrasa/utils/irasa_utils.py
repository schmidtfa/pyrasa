"""Utilities for signal decompositon using IRASA"""

import fractions
from collections.abc import Callable
from copy import copy

import numpy as np
import scipy.signal as dsp

from pyrasa.utils.types import IrasaFun


def _gen_irasa(
    data: np.ndarray,
    orig_spectrum: np.ndarray,
    fs: int,
    irasa_fun: IrasaFun,
    hset: np.ndarray,
    time: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate original, aperiodic, and periodic spectra using the IRASA algorithm.

    This function implements the IRASA (Irregular Resampling Auto-Spectral Analysis) algorithm
    to decompose a power or cross-spectral density into its periodic and aperiodic components.

    Parameters
    ----------
    data : np.ndarray
        The input time-series data, typically with shape (n_channels, n_times) or similar.
    orig_spectrum : np.ndarray
        The original power spectral density from which periodic and aperiodic components are to be extracted.
    fs : int
        The sampling frequency of the input data in Hz.
    irasa_fun : IrasaFun
        A custom function used to compute power spectral densities. This function should
        take resampled data and return the corresponding spectrum.
    hset : np.ndarray
        An array of up/downsampling factors (e.g., [1.1, 1.2, 1.3, ...]) used in the IRASA algorithm.
    time : np.ndarray | None, optional
        The time vector associated with the original data. This is only necessary if the IRASA function
        requires the time stamps of the original data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - `orig_spectrum` (np.ndarray): The original spectrum provided as input.
        - `aperiodic_spectrum` (np.ndarray): The median of the geometric mean of up/downsampled spectra,
        representing the aperiodic component.
        - `periodic_spectrum` (np.ndarray): The difference between the original and the aperiodic spectrum,
        representing the periodic component.

    Notes
    -----
    This implementation of the IRASA algorithm is based on the `yasa.irasa` function from (Vallat & Walker, 2021).
    The IRASA algorithm involves upsampling and downsampling the time-series data by a set of factors (`hset`),
    calculating the power spectra of these resampled data, and then taking the geometric mean of the upsampled
    and downsampled spectra to isolate the aperiodic component.

    References
    ----------
    [1] Vallat, Raphael, and Matthew P. Walker. “An open-source, high-performance tool for automated sleep staging.”
        Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092
    """

    spectra = np.zeros((len(hset), *orig_spectrum.shape))
    for i, h in enumerate(hset):
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator

        # Much faster than FFT-based resampling
        data_up = dsp.resample_poly(data, up, down, axis=-1)
        data_down = dsp.resample_poly(data, down, up, axis=-1)

        # Calculate an up/downsampled version of the PSD using same params as original
        spectrum_up = irasa_fun(data=data_up, fs=int(fs * h), h=h, time_orig=time, up_down='up')
        spectrum_dw = irasa_fun(data=data_down, fs=int(fs / h), h=h, time_orig=time, up_down='down')

        # geometric mean between up and downsampled
        spectra[i, :, :] = np.sqrt(spectrum_up * spectrum_dw)

    aperiodic_spectrum = np.median(spectra, axis=0)
    periodic_spectrum = orig_spectrum - aperiodic_spectrum
    return orig_spectrum, aperiodic_spectrum, periodic_spectrum


def _crop_data(
    band: list | tuple,
    freqs: np.ndarray,
    psd_aperiodic: np.ndarray,
    psd_periodic: np.ndarray,
    psd: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Utility function to crop spectra to a defined frequency range"""

    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=axis)
    psd_periodic = np.compress(~mask_freqs, psd_periodic, axis=axis)
    psd = np.compress(~mask_freqs, psd, axis=axis)

    return freqs, psd_aperiodic, psd_periodic, psd


def _get_windows(
    nperseg: int, dpss_settings: dict, win_func: Callable, win_func_kwargs: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a window function used for tapering"""

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
    """Check if the input parameters for irasa are specified correctly"""

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
        f'which is higher than the Nyquist frequency for your data of {nyquist}Hz, \n'
        'try to either lower the upper bound for the hset or decrease the upper band limit, when running IRASA.'
    )

    filter_settings: list[float] = list(irasa_params['filter_settings'])
    if filter_settings[0] is None:
        filter_settings[0] = band_evaluated[0]
    if filter_settings[1] is None:
        filter_settings[1] = band_evaluated[1]

    assert np.logical_and(band_evaluated[0] >= filter_settings[0], band_evaluated[1] <= filter_settings[1]), (
        f'You run IRASA in a frequency range from '
        f'{np.round(band_evaluated[0], irasa_params["hset_accuracy"])} - '
        f'{np.round(band_evaluated[1], irasa_params["hset_accuracy"])}Hz. \n'
        'Your settings specified in "filter_settings" indicate that you have a pass band from '
        f'{np.round(filter_settings[0], irasa_params["hset_accuracy"])} - '
        f'{np.round(filter_settings[1], irasa_params["hset_accuracy"])}Hz. \n'
        'This means that your evaluated range likely contains filter artifacts. \n'
        'Either change your filter settings, adjust hset or the parameter "band" accordingly. \n'
        f'You want to make sure that the lower band limit divided by the upper bound of the hset '
        f'> {np.round(filter_settings[0], irasa_params["hset_accuracy"])} \n'
        'and that upper band limit times the upper bound of the hset < '
        f'{np.round(filter_settings[1], irasa_params["hset_accuracy"])}'
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
    """Compute power spectral densities via scipy.signal.welch"""

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
    nfft: int,
    hop: int,
    win_duration: float,
    dpss_settings: dict,
    win_kwargs: dict,
    h: float = 1.0,
    up_down: str | None = None,
    time_orig: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrograms via scipy.signal.stft"""

    nperseg = int(np.floor(fs * win_duration))

    if up_down == 'up':
        hop = int(hop * h)

    if up_down == 'down':
        hop = int(hop / h)

    win, ratios = _get_windows(nperseg, dpss_settings, **win_kwargs)

    sgramms = []
    for cur_win in win:
        freq, time, sgramm = dsp.stft(
            x, nfft=nfft, nperseg=nperseg, noverlap=nperseg - hop, fs=fs, window=cur_win, scaling='psd'
        )
        sgramm = np.abs(sgramm) ** 2
        sgramms.append(sgramm)

    if ratios is None:
        sgramm = np.mean(sgramms, axis=0)
    else:
        weighted_sgramms = [ratios[ix] * cur_sgramm for ix, cur_sgramm in enumerate(sgramms)]
        sgramm = np.sum(weighted_sgramms, axis=0) / np.sum(ratios)

    if time_orig is not None:
        sgramm = sgramm[:, :, : time_orig.shape[-1]]

    sgramm = np.squeeze(sgramm)

    return freq, time, sgramm
