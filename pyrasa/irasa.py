"""Functions to compute IRASA."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.signal as dsp

from pyrasa.utils.irasa_spectrum import IrasaSpectrum
from pyrasa.utils.irasa_tf_spectrum import IrasaTfSpectrum

# from scipy.stats.mstats import gmean
from pyrasa.utils.irasa_utils import (
    _check_irasa_settings,
    _compute_psd_welch,
    _compute_sgramm,
    _crop_data,
    _gen_irasa,
)

if TYPE_CHECKING:
    from pyrasa.utils.types import IrasaSprintKwargsTyped


# %% irasa
def irasa(
    data: np.ndarray,
    fs: int,
    band: tuple[float, float],
    psd_kwargs: dict,
    ch_names: np.ndarray | None = None,
    win_func: Callable = dsp.windows.hann,
    win_func_kwargs: dict | None = None,
    dpss_settings_time_bandwidth: float = 2.0,
    dpss_settings_low_bias: bool = True,
    dpss_eigenvalue_weighting: bool = True,
    filter_settings: tuple[float | None, float | None] = (None, None),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    hset_accuracy: int = 4,
) -> IrasaSpectrum:
    """
    Computes the aperiodic and periodic components of the power spectrum from a time series using the
    Irregular Resampling Autocorrelation (IRASA) algorithm.

    The IRASA algorithm allows for the decomposition of neural signals into fractal (aperiodic) and
    oscillatory (periodic) components, providing insight into the underlying dynamics of the data.

    Parameters
    ----------
    data : np.ndarray
        Time series data, where the shape is expected to be either (Samples,) or (Channels, Samples).
    fs : int
        Sampling frequency of the data in Hz.
    band : tuple[float, float]
        The frequency range (lower and upper bounds in Hz) over which to compute the spectra.
    psd_kwargs : dict
        Keyword arguments to be passed to the `scipy.signal.welch` function for PSD estimation.
    ch_names : np.ndarray | None, optional
        Channel names associated with the data, if available. Default is None.
    win_func : Callable, optional
        Window function to be used in Welch's method. Default is `dsp.windows.hann`.
    win_func_kwargs : dict | None, optional
        Additional keyword arguments for the window function. Default is None.
    dpss_settings_time_bandwidth : float, optional
        Time-bandwidth product for the DPSS windows if used. Default is 2.0.
    dpss_settings_low_bias : bool, optional
        Keep only tapers with eigenvalues > 0.9. Default is True.
    dpss_eigenvalue_weighting : bool, optional
        Whether or not to apply eigenvalue weighting in DPSS. If True, spectral estimates weighted by
        the concentration ratio of their respective tapers before combining. Default is True.
    filter_settings : tuple[float | None, float | None], optional
        Cutoff frequencies for highpass and lowpass filtering to avoid artifacts in the evaluated frequency range.
        Default is (None, None).
    hset_info : tuple[float, float, float], optional
        Tuple specifying the range of the resampling factors as (min, max, step). Default is (1.05, 2.0, 0.05).
    hset_accuracy : int, optional
        Decimal precision for the resampling factors. Default is 4.

    Returns
    -------
    IrasaSpectrum
        An object containing the following attributes:
            - freqs: np.ndarray
                Frequencies corresponding to the computed spectra.
            - raw_spectrum: np.ndarray
                The raw power spectrum.
            - aperiodic: np.ndarray
                The aperiodic (fractal) component of the spectrum.
            - periodic: np.ndarray
                The periodic (oscillatory) component of the spectrum.
            - ch_names: np.ndarray
                Channel names if provided.

    Notes
    -----
    This function provides fine-grained control over the IRASA parameters. For users working with MNE-Python,
    the `irasa_raw` and `irasa_epochs` functions from `pyrasa.irasa_mne` are recommended, as they handle
    additional preprocessing steps.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum
    of Neurophysiological Signal. Brain Topography, 29(1), 13–26. https://doi.org/10.1007/s10548-015-0448-0
    """

    # set parameters
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
    hset = [h for h in hset if h % 1 != 0]  # filter integers

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

    freq, psd_aperiodic, psd_periodic, psd = _crop_data(band, freq, psd_aperiodic, psd_periodic, psd, axis=-1)

    return IrasaSpectrum(
        freqs=freq,
        raw_spectrum=psd,
        aperiodic=psd_aperiodic,
        periodic=psd_periodic,
        ch_names=ch_names,
    )


# irasa sprint
def irasa_sprint(  # noqa PLR0915 C901
    data: np.ndarray,
    fs: int,
    ch_names: np.ndarray | None = None,
    band: tuple[float, float] = (1.0, 100.0),
    win_duration: float = 0.4,
    overlap_fraction: float = 0.90,
    win_func: Callable = dsp.windows.hann,
    win_func_kwargs: dict | None = None,
    dpss_settings_time_bandwidth: float = 2.0,
    dpss_settings_low_bias: bool = True,
    dpss_eigenvalue_weighting: bool = True,
    filter_settings: tuple[float | None, float | None] = (None, None),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    hset_accuracy: int = 4,
) -> IrasaTfSpectrum:
    """
    Computes time-resolved aperiodic and periodic components of the power spectrum from a time series
    using the Irregular Resampling Autocorrelation (IRASA) algorithm.

    This function is useful for analyzing how the aperiodic and periodic components of the power spectrum
    change over time, providing a time-frequency decomposition of the signal.

    Parameters
    ----------
    data : np.ndarray
        Time series data, where the shape is expected to be either (Samples,) or (Channels, Samples).
    fs : int
        Sampling frequency of the data in Hz.
    ch_names : np.ndarray | None, optional
        Channel names associated with the data, if available. Default is None.
    band : tuple[float, float], optional
        The frequency range (lower and upper bounds in Hz) over which to compute the spectra. Default is (1.0, 100.0).
    win_duration : float, optional
        Duration of the window in seconds used for the short-time Fourier transforms (STFTs). Default is 0.4 seconds.
    overlap_fraction : int, optional
        The overlap between the STFT sliding windows as fraction. Default is .99 of the windows.
    win_func : Callable, optional
        Window function to be used in computing the time frequency spectrum. Default is `dsp.windows.hann`.
    win_func_kwargs : dict | None, optional
        Additional keyword arguments for the window function. Default is None.
    dpss_settings_time_bandwidth : float, optional
        Time-bandwidth product for the DPSS windows if used. Default is 2.0.
    dpss_settings_low_bias : bool, optional
        Keep only tapers with eigenvalues > 0.9. Default is True.
    dpss_eigenvalue_weighting : bool, optional
        Whether or not to apply eigenvalue weighting in DPSS. If True, spectral estimates weighted by
        the concentration ratio of their respective tapers before combining. Default is True.
    filter_settings : tuple[float | None, float | None], optional
        Cutoff frequencies for highpass and lowpass filtering to avoid artifacts in the evaluated frequency range.
        Default is (None, None).
    hset_info : tuple[float, float, float], optional
        Tuple specifying the range of the resampling factors as (min, max, step). Default is (1.05, 2.0, 0.05).
    hset_accuracy : int, optional
        Decimal precision for the resampling factors. Default is 4.

    Returns
    -------
    IrasaTfSpectrum
        An object containing the following attributes:
            - freqs: np.ndarray
                Frequencies corresponding to the computed spectra.
            - time: np.ndarray
                Time bins in seconds associated with the (a-)periodic spectra.
            - raw_spectrum: np.ndarray
                The raw time-frequency power spectrum.
            - aperiodic: np.ndarray
                The aperiodic (fractal) component of the spectrum.
            - periodic: np.ndarray
                The periodic (oscillatory) component of the spectrum.
            - ch_names: np.ndarray
                Channel names if provided.

    Notes
    -----
    This function performs a time-frequency decomposition of the input data, allowing for a time-resolved analysis
    of the periodic and aperiodic components of the signal. The STFT is computed for each time window, and IRASA
    is applied to separate the spectral components.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum of
    Neurophysiological Signal. Brain Topography, 29(1), 13–26. https://doi.org/10.1007/s10548-015-0448-0
    """

    # set parameters
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
    }

    _check_irasa_settings(irasa_params=irasa_params, hset_info=hset_info)

    hset = np.round(np.arange(*hset_info), hset_accuracy)
    hset = [h for h in hset if h % 1 != 0]  # filter integers

    win_kwargs = {'win_func': win_func, 'win_func_kwargs': win_func_kwargs}
    dpss_settings = {
        'time_bandwidth': dpss_settings_time_bandwidth,
        'low_bias': dpss_settings_low_bias,
        'eigenvalue_weighting': dpss_eigenvalue_weighting,
    }

    nfft = int(2 ** np.ceil(np.log2(np.max(hset) * win_duration * fs)))
    hop = int((1 - overlap_fraction) * win_duration * fs)
    # hop = int((1 - overlap_fraction) * nfft)
    irasa_kwargs: IrasaSprintKwargsTyped = {
        'nfft': nfft,
        'hop': hop,
        'win_duration': win_duration,
        'dpss_settings': dpss_settings,
        'win_kwargs': win_kwargs,
    }

    def _local_irasa_fun(
        data: np.ndarray,
        fs: int,
        h: float,
        up_down: str | None,
        time_orig: np.ndarray | None = None,
    ) -> np.ndarray:
        return _compute_sgramm(data, fs, h=h, up_down=up_down, time_orig=time_orig, **irasa_kwargs)[2]

    # get time and frequency info
    freq, time, sgramm = _compute_sgramm(data, fs, **irasa_kwargs)

    sgramm, sgramm_aperiodic, sgramm_periodic = _gen_irasa(
        data=data,
        orig_spectrum=sgramm,
        fs=fs,
        irasa_fun=_local_irasa_fun,
        hset=hset,
        time=time,
    )
    single_ch_dim = 2
    sgramm_aperiodic = (
        sgramm_aperiodic[np.newaxis, :, :] if sgramm_aperiodic.ndim == single_ch_dim else sgramm_aperiodic
    )
    sgramm_periodic = sgramm_periodic[np.newaxis, :, :] if sgramm_periodic.ndim == single_ch_dim else sgramm_periodic

    freq, sgramm_aperiodic, sgramm_periodic, sgramm = _crop_data(
        band, freq, sgramm_aperiodic, sgramm_periodic, sgramm, axis=1
    )

    # adjust time info (i.e. cut the padded stuff)
    tmax = data.shape[1] / fs
    t_mask = np.logical_and(time >= 0, time < tmax)
    freq_mask = freq > (1 / win_duration)  # mask rayleigh

    return IrasaTfSpectrum(
        freqs=freq[freq_mask],
        time=time[t_mask],
        raw_spectrum=sgramm,
        periodic=sgramm_periodic[:, :, t_mask][:, freq_mask, :],
        aperiodic=sgramm_aperiodic[:, :, t_mask][:, freq_mask, :],
        ch_names=ch_names,
    )
