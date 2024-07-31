"""Functions to compute the IRASA algorithm."""

import fractions
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.signal as dsp

# from scipy.stats.mstats import gmean
from pyrasa.utils.irasa_utils import (
    _check_irasa_settings,
    _compute_psd_welch,
    _compute_sgramm,
    _crop_data,  # _find_nearest, _gen_time_from_sft, _get_windows,
)
from pyrasa.utils.types import IrasaFun

if TYPE_CHECKING:
    from pyrasa.utils.input_classes import IrasaSprintKwargsTyped


# TODO: Port to Cython
def _gen_irasa(
    data: np.ndarray,
    orig_spectrum: np.ndarray,
    fs: int,
    irasa_fun: IrasaFun,
    hset: np.ndarray,
    time: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the IRASA algorithm using a custom function to compute power/cross-spectral density.

    This function returns an "original", "periodic" and "aperiodic spectrum". This implementation
    of the IRASA algorithm is based on the yasa.irasa function in (Vallat & Walker, 2021).

    Parameters
    ----------
    data : np.ndarray
        The input data.
    orig_spectrum : np.ndarray
        The original power spectrum.
    fs : int
        The sampling frequency.
    irasa_fun : IrasaFun
        The function to compute the power/cross-spectral density.
    hset : np.ndarray
        Array of up/downsampling factors.
    time : np.ndarray | None, optional
        Time array. Default is None.

    Returns
    -------
    tuple of np.ndarray
        The original, aperiodic, and periodic spectra.

    References
    ----------
    Vallat, Raphael, and Matthew P. Walker. “An open-source, high-performance tool for automated sleep staging.”
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

        # Geometric mean between up and downsampled
        if spectra.ndim == 2:  # noqa PLR2004
            spectra[i, :] = np.sqrt(spectrum_up * spectrum_dw)
        if spectra.ndim == 3:  # noqa PLR2004
            spectra[i, :, :] = np.sqrt(spectrum_up * spectrum_dw)

    aperiodic_spectrum = np.median(spectra, axis=0)
    periodic_spectrum = orig_spectrum - aperiodic_spectrum
    return orig_spectrum, aperiodic_spectrum, periodic_spectrum


# %% irasa
def irasa(
    data: np.ndarray,
    fs: int,
    band: tuple[float, float],
    psd_kwargs: dict,
    win_func: Callable = dsp.windows.hann,
    win_func_kwargs: dict | None = None,
    dpss_settings_time_bandwidth: float = 2.0,
    dpss_settings_low_bias: bool = True,
    dpss_eigenvalue_weighting: bool = True,
    filter_settings: tuple[float | None, float | None] = (None, None),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    hset_accuracy: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate aperiodic and periodic power spectra from a time series using the IRASA algorithm.

    This function gives you maximal control over all parameters, so it is up to you to set things up properly.
    If you have preprocessed your data in mne python, it is recommended to use the
    irasa_raw or irasa_epochs functions from `pyrasa.irasa_mne`, as they directly work on your
    `mne.io.BaseRaw` and `mne.io.BaseEpochs` classes and take care of the necessary checks.

    Parameters
    ----------
    data : np.ndarray
        The timeseries data used to extract aperiodic and periodic power spectra.
    fs : int
        The sampling frequency of the data.
    band : tuple of float
        The lower and upper band of the frequency range used to extract (a-)periodic spectra.
    psd_kwargs : dict
        A dictionary containing all the keyword arguments that are passed onto `scipy.signal.welch`.
    win_func : Callable, optional
        Window function to use. Defaults to `scipy.signal.windows.hann`.
    win_func_kwargs : dict | None, optional
        Additional keyword arguments for the window function. Defaults to None.
    dpss_settings_time_bandwidth : float, optional
        Time-bandwidth product for the DPSS window. Defaults to 2.0.
    dpss_settings_low_bias : bool, optional
        If True, provides a low-bias DPSS window. Defaults to True.
    dpss_eigenvalue_weighting : bool, optional
        If True, applies eigenvalue weighting for the DPSS window. Defaults to True.
    filter_settings : tuple of float | None, optional
        The cut-off frequencies for the high-pass and low-pass filters. Defaults to (None, None).
    hset_info : tuple of float, optional
        Contains information about the range of the up/downsampling factors. Defaults to (1.05, 2.0, 0.05).
    hset_accuracy : int, optional
        Floating point accuracy for the up/downsampling factor of the signal. Defaults to 4.

    Returns
    -------
    tuple of np.ndarray
        The frequencies, aperiodic component, and periodic component of the data.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum of
    Neurophysiological Signal. Brain Topography, 29(1), 13–26. https://doi.org/10.1007/s10548-015-0448-0
    """
    if win_func_kwargs is None:
        win_func_kwargs = {}

    # Minimal safety checks
    if data.ndim == 1:
        data = data[np.newaxis, :]
    assert data.ndim == 2, 'Data shape needs to be either of shape (Channels, Samples) or (Samples, ).'  # noqa PLR2004

    irasa_params = {
        'data': data,
        'fs': fs,
        'band': band,
        'filter_settings': filter_settings,
        'hset_accuracy': hset_accuracy,
    }

    _check_irasa_settings(irasa_params=irasa_params, hset_info=hset_info)

    hset = np.round(np.arange(*hset_info), hset_accuracy)

    win_kwargs = {'win_func': win_func, 'win_func_kwargs': win_func_kwargs}
    dpss_settings = {
        'time_bandwidth': dpss_settings_time_bandwidth,
        'low_bias': dpss_settings_low_bias,
        'eigenvalue_weighting': dpss_eigenvalue_weighting,
    }

    def _local_irasa_fun(
        data: np.ndarray,
        fs: int,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        return _compute_psd_welch(
            data,
            fs=fs,
            nperseg=psd_kwargs.get('nperseg'),
            win_kwargs=win_kwargs,
            dpss_settings=dpss_settings,
            noverlap=psd_kwargs.get('noverlap'),
            nfft=psd_kwargs.get('nfft'),
        )[1]

    freq, psd = _compute_psd_welch(
        data,
        fs=fs,
        nperseg=psd_kwargs.get('nperseg'),
        win_kwargs=win_kwargs,
        dpss_settings=dpss_settings,
        noverlap=psd_kwargs.get('noverlap'),
        nfft=psd_kwargs.get('nfft'),
    )

    psd, psd_aperiodic, psd_periodic = _gen_irasa(
        data=np.squeeze(data), orig_spectrum=psd, fs=fs, irasa_fun=_local_irasa_fun, hset=hset
    )

    freq, psd_aperiodic, psd_periodic = _crop_data(band, freq, psd_aperiodic, psd_periodic, axis=-1)

    return freq, psd_aperiodic, psd_periodic


# irasa sprint
def irasa_sprint(  # noqa PLR0915 C901
    data: np.ndarray,
    fs: int,
    band: tuple[float, float] = (1.0, 100.0),
    freq_res: float = 0.5,
    win_duration: float = 0.4,
    hop: int = 10,
    win_func: Callable = dsp.windows.hann,
    win_func_kwargs: dict | None = None,
    dpss_settings_time_bandwidth: float = 2.0,
    dpss_settings_low_bias: bool = True,
    dpss_eigenvalue_weighting: bool = True,
    filter_settings: tuple[float | None, float | None] = (None, None),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    hset_accuracy: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate aperiodic from periodic power spectra using the IRASA algorithm in a time-resolved manner.

    Parameters
    ----------
    data : np.ndarray
        The timeseries data used to extract aperiodic and periodic power spectra.
    fs : int
        The sampling frequency of the data.
    band : tuple of float, optional
        The lower and upper band of the frequency range used to extract (a-)periodic spectra. Defaults to (1.0, 100.0).
    freq_res : float, optional
        The desired frequency resolution in Hz. Defaults to 0.5.
    win_duration : float, optional
        The time width of the window in seconds used to calculate the STFFTs. Defaults to 0.4.
    hop : int, optional
        Time increment in signal samples for the sliding window. Defaults to 10.
    win_func : Callable, optional
        The desired window function. Defaults to `scipy.signal.windows.hann`.
    win_func_kwargs : dict | None, optional
        A dictionary containing keyword arguments passed to `win_func`. Defaults to None.
    dpss_settings_time_bandwidth : float, optional
        Time-bandwidth product for the DPSS window. Defaults to 2.0.
    dpss_settings_low_bias : bool, optional
        If True, provides a low-bias DPSS window. Defaults to True.
    dpss_eigenvalue_weighting : bool, optional
        If True, applies eigenvalue weighting for the DPSS window. Defaults to True.
    filter_settings : tuple of float | None, optional
        The cut-off frequencies for the high-pass and low-pass filters. Defaults to (None, None).
    hset_info : tuple of float, optional
        Contains information about the range of the up/downsampling factors. Defaults to (1.05, 2.0, 0.05).
    hset_accuracy : int, optional
        Floating point accuracy for the up/downsampling factor of the signal. Defaults to 4.

    Returns
    -------
    tuple of np.ndarray
        The aperiodic component, periodic component, frequencies, and
        time bins associated with the (a-)periodic spectra.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum of
    Neurophysiological Signal. Brain Topography, 29(1), 13–26. https://doi.org/10.1007/s10548-015-0448-0
    """
    if win_func_kwargs is None:
        win_func_kwargs = {}

    # Safety checks
    assert isinstance(data, np.ndarray), 'Data should be a numpy array.'
    assert data.ndim == 2, 'Data shape needs to be of shape (Channels, Samples).'  # noqa PLR2004

    irasa_params = {
        'data': data,
        'fs': fs,
        'band': band,
        'filter_settings': filter_settings,
    }

    _check_irasa_settings(irasa_params=irasa_params, hset_info=hset_info)

    hset = np.round(np.arange(*hset_info), hset_accuracy)

    mfft = int(fs / freq_res)
    win_kwargs = {'win_func': win_func, 'win_func_kwargs': win_func_kwargs}
    dpss_settings = {
        'time_bandwidth': dpss_settings_time_bandwidth,
        'low_bias': dpss_settings_low_bias,
        'eigenvalue_weighting': dpss_eigenvalue_weighting,
    }

    irasa_kwargs: IrasaSprintKwargsTyped = {
        'mfft': mfft,
        'hop': hop,
        'win_duration': win_duration,
        'dpss_settings': dpss_settings,
        'win_kwargs': win_kwargs,
    }

    def _local_irasa_fun(
        data: np.ndarray,
        fs: int,
        h: int | None,
        up_down: str | None,
        time_orig: np.ndarray | None = None,
    ) -> np.ndarray:
        return _compute_sgramm(data, fs, h=h, up_down=up_down, time_orig=time_orig, **irasa_kwargs)[2]

    # Get time and frequency info
    freq, time, sgramm = _compute_sgramm(data, fs, **irasa_kwargs)

    sgramm, sgramm_aperiodic, sgramm_periodic = _gen_irasa(
        data=data,
        orig_spectrum=sgramm,
        fs=fs,
        irasa_fun=_local_irasa_fun,
        hset=hset,
        time=time,
    )

    # NOTE: we need to transpose the data as crop_data extracts stuff from the last axis
    freq, sgramm_aperiodic, sgramm_periodic = _crop_data(band, freq, sgramm_aperiodic, sgramm_periodic, axis=0)

    # Adjust time info (i.e. cut the padded stuff)
    tmax = data.shape[1] / fs
    t_mask = np.logical_and(time >= 0, time < tmax)

    return sgramm_aperiodic[:, t_mask], sgramm_periodic[:, t_mask], freq, time[t_mask]
