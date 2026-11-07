"""Functions to compute IRASA."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

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
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    detrend: str | Callable | Literal[False] = 'constant',
    scaling: str = 'density',
    average: str = 'mean',
    win_func: Callable = dsp.windows.hann,
    win_func_kwargs: dict | None = None,
    dpss_settings_time_bandwidth: float = 2.0,
    dpss_settings_low_bias: bool = True,
    dpss_eigenvalue_weighting: bool = True,
    ch_names: np.ndarray | None = None,
    filter_settings: tuple[float | None, float | None] = (None, None),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    hset_accuracy: int = 4,
) -> IrasaSpectrum:
    """
    Decompose the power spectrum into aperiodic and periodic components using IRASA.

    Apply Irregular Resampling Auto-Spectral Analysis (IRASA; Wen & Liu, 2016)
    to separate the power spectrum of a time series into an aperiodic
    (fractal) and a periodic (oscillatory) component.

    IRASA resamples the signal using pairs of non-integer factors (h and 1/h).
    This shifts oscillatory peaks in frequency while largely preserving the
    aperiodic spectral structure. For each resampling pair, the geometric
    mean of the resulting power spectra is calculated. The median across
    resampling pairs provides an estimate of the aperiodic spectrum.
    Subtracting this estimate from the original power spectrum yields the
    periodic component.

    Power spectral densities are estimated using Welch's method, with
    optional multitaper estimation using discrete prolate spheroidal
    sequences (DPSS).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,) or (n_channels, n_samples)
        Time-series data to decompose. One-dimensional input is treated
        as a single-channel signal.
    fs : int
        Sampling frequency of the data, in Hz.
    band : tuple[float, float]
        Lower and upper frequency bounds of the returned spectra, in Hz.
        The requested range must be compatible with the sampling frequency,
        resampling factors, and specified filter settings.
    nperseg : int | None, optional
        Number of samples per segment used for Welch's spectral estimation.
        If None, the segment length is set to the length of the shortest
        resampled signal. Explicitly specified values must not exceed
        this length.
    noverlap : int | None, optional
        Number of samples overlapping between consecutive Welch segments.
        If None, half of `nperseg` is used.
    nfft : int | None, optional
        Number of FFT points used for spectral estimation. If None,
        `nperseg` is used. Values larger than `nperseg` provide
        zero-padding.
    detrend : str | Callable | Literal[False], default 'constant'
        Detrending applied to each segment before spectral estimation.
        A string specifies the detrending method, a callable defines
        a custom detrending function, and False disables detrending.
    scaling : {'density', 'spectrum'}, default 'density'
        Scaling of the spectral estimates. 'density' returns power
        spectral density (power per Hz), whereas 'spectrum' returns
        power spectrum values.
    average : {'mean', 'median'}, default 'mean'
        Method used to average Welch periodograms across segments.
    win_func : Callable, default scipy.signal.windows.hann
        Window function used for spectral estimation. DPSS windows
        can be selected for multitaper estimation.
    win_func_kwargs : dict | None, optional
        Additional keyword arguments passed to the window function.
    dpss_settings_time_bandwidth : float, default 2.0
        Time-bandwidth product used when DPSS windows are selected.
    dpss_settings_low_bias : bool, default True
        Whether to retain only DPSS tapers with spectral concentration
        ratios greater than 0.9.
    dpss_eigenvalue_weighting : bool, default True
        Whether to weight the spectral estimates of individual DPSS
        tapers by their spectral concentration ratios.
    ch_names : np.ndarray | None, optional
        Channel names corresponding to the input data.
    filter_settings : tuple[float | None, float | None], default (None, None)
        High-pass and low-pass filter cutoff frequencies of the input
        data, in Hz. Used to verify that the frequency range required
        for IRASA is compatible with the signal's passband. No filtering
        is performed by this function.
    hset_info : tuple[float, float, float], default (1.05, 2.0, 0.05)
        Minimum, maximum, and step size of the resampling factors.
        Factors are generated using `np.arange`, excluding the upper
        bound and integer-valued factors.
    hset_accuracy : int, default 4
        Number of decimal places used to round the resampling factors.

    Returns
    -------
    IrasaSpectrum
        Object containing the following attributes:

        - `freqs`: Frequency values of the returned spectra, in Hz.
        - `raw_spectrum`: Original power spectrum, with shape
        (n_channels, n_freqs).
        - `aperiodic`: IRASA-estimated aperiodic component, with shape
        (n_channels, n_freqs).
        - `periodic`: Periodic component, with shape
        (n_channels, n_freqs).
        - `ch_names`: Channel names, if provided.

    Notes
    -----
    The periodic component is calculated as the difference between the
    original and aperiodic spectra and can therefore contain negative values.

    The returned spectra are restricted to the frequency bins within `band`.

    For MNE-Python objects, the `irasa_raw` and `irasa_epochs` functions
    provide convenient interfaces that extract measurement information
    and return MNE-compatible spectral objects.

    See Also
    --------
    irasa_sprint
    pyrasa.irasa_mne.irasa_raw
    pyrasa.irasa_mne.irasa_epochs

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components
    in the Power Spectrum of Neurophysiological Signal.
    Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0
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
    hset = np.array([h for h in hset if h % 1 != 0])  # filter integers

    min_samples = int(np.ceil(data.shape[-1] / hset.max()))

    if nperseg is None:
        nperseg = min_samples
    elif nperseg > min_samples:
        raise ValueError('nperseg exceeds the length of the shortest resampled signal.')
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
            nperseg=nperseg,
            noverlap=noverlap,
            average=average,
            nfft=nfft,
            detrend=detrend,
            scaling=scaling,
            win_kwargs=win_kwargs,
            dpss_settings=dpss_settings,
        )[1]

    freq, psd = _compute_psd_welch(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        average=average,
        scaling=scaling,
        win_kwargs=win_kwargs,
        dpss_settings=dpss_settings,
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
    Compute time-resolved aperiodic and periodic spectra using IRASA.

    Apply Irregular Resampling Auto-Spectral Analysis (IRASA; Wen & Liu, 2016)
    to decompose a time series into time-varying aperiodic (fractal) and
    periodic (oscillatory) spectral components.

    Unlike `irasa`, which estimates a power spectrum across a recording,
    this function uses short-time Fourier transforms (STFTs) to obtain
    time-frequency representations of the signal. IRASA is then applied
    to separate the aperiodic and periodic components at each time point.

    For each non-integer resampling factor (h), the geometric mean of
    the upsampled and downsampled spectrograms is calculated. The median
    across resampling factors estimates the aperiodic spectrogram, which
    is subtracted from the original spectrogram to obtain the periodic
    component.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,) or (n_channels, n_samples)
        Time-series data to decompose. One-dimensional input is treated
        as a single-channel signal.
    fs : int
        Sampling frequency of the data, in Hz.
    ch_names : np.ndarray | None, optional
        Channel names corresponding to the input data.
    band : tuple[float, float], default (1.0, 100.0)
        Lower and upper frequency bounds of the returned spectrograms,
        in Hz. Frequencies at or below the Rayleigh frequency
        (1 / win_duration) are additionally excluded.
    win_duration : float, default 0.4
        Duration of each STFT analysis window, in seconds.
    overlap_fraction : float, default 0.90
        Fraction of overlap between consecutive STFT windows.
        The nominal hop size in samples is calculated as
        `int((1 - overlap_fraction) * win_duration * fs)`.
        The overlap must allow a valid, positive hop size.
    win_func : Callable, default scipy.signal.windows.hann
        Window function used for time-frequency estimation. DPSS
        windows can be selected for multitaper estimation.
    win_func_kwargs : dict | None, optional
        Additional keyword arguments passed to the window function.
    dpss_settings_time_bandwidth : float, default 2.0
        Time-bandwidth product used when DPSS windows are selected.
    dpss_settings_low_bias : bool, default True
        Whether to retain only DPSS tapers with spectral concentration
        ratios greater than 0.9.
    dpss_eigenvalue_weighting : bool, default True
        Whether to weight the spectral estimates of individual DPSS
        tapers by their spectral concentration ratios.
    filter_settings : tuple[float | None, float | None], default (None, None)
        High-pass and low-pass filter cutoff frequencies of the input
        data, in Hz. Used to verify that the frequency range required
        for IRASA is compatible with the signal's passband. No filtering
        is performed by this function.
    hset_info : tuple[float, float, float], default (1.05, 2.0, 0.05)
        Minimum, maximum, and step size of the resampling factors.
        Factors are generated using `np.arange`, excluding the upper
        bound and integer-valued factors.
    hset_accuracy : int, default 4
        Number of decimal places used to round the resampling factors.

    Returns
    -------
    IrasaTfSpectrum
        Object containing the following attributes:

        - `freqs`: Frequency values of the returned spectrograms, in Hz.
        - `time`: Time points corresponding to the STFT window centers,
        in seconds.
        - `raw_spectrum`: Original time-frequency power spectrum, with
        shape (n_channels, n_freqs, n_times).
        - `aperiodic`: IRASA-estimated aperiodic spectrogram, with
        shape (n_channels, n_freqs, n_times).
        - `periodic`: Periodic spectrogram, with shape
        (n_channels, n_freqs, n_times).
        - `ch_names`: Channel names, if provided.

    Notes
    -----
    This function performs a time-frequency decomposition of the input data, allowing for a time-resolved analysis
    of the periodic and aperiodic components of the signal. The STFT is computed for each time window, and IRASA
    is applied to separate the spectral components.

    See Also
    --------
    irasa

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components
    in the Power Spectrum of Neurophysiological Signal.
    Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0
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
    hset = np.array([h for h in hset if h % 1 != 0])  # filter integers

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
    sgramm = sgramm[np.newaxis, :, :] if sgramm.ndim == single_ch_dim else sgramm

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
        raw_spectrum=sgramm[:, :, t_mask][:, freq_mask, :],
        periodic=sgramm_periodic[:, :, t_mask][:, freq_mask, :],
        aperiodic=sgramm_aperiodic[:, :, t_mask][:, freq_mask, :],
        ch_names=ch_names,
    )
