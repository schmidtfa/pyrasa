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
    This function is implementing the IRASA algorithm using a custom function to
    compute a power/cross-spectral density and returns an "original", "periodic" and "aperiodic spectrum".
    This implementation of the IRASA algorithm is based on the yasa.irasa function in (Vallat & Walker, 2021).

    [1] Vallat, Raphael, and Matthew P. Walker. “An open-source,
    high-performance tool for automated sleep staging.”
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
        # be aware of the input dimensions
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
    This function can be used to generate aperiodic and periodic power spectra from a time series
    using the IRASA algorithm (Wen & Liu, 2016).

    This function gives you maximal control over all parameters so its up to you set things up properly.

    If you have preprocessed your data in mne python we recommend that you use the
    irasa_raw or irasa_epochs functions from `pyrasa.irasa_mne`, as they directly work on your
    `mne.io.BaseRaw` and `mne.io.BaseEpochs` classes and take care of the necessary checks.

    Parameters
    ----------
    data : :py:class:˚numpy.ndarray˚
        The timeseries data used to extract aperiodic and periodic power spectra.
    fs : int
        The sampling frequency of the data. Can be omitted if data is :py:class:˚mne.io.BaseRaw˚.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract (a-)periodic spectra.
    psd_kwargs : dict
        A dictionary containing all the keyword arguments that are passed onto `scipy.signal.welch`.
    filter_settings : tuple
        A tuple containing the cut-off of the High- and Lowpass filter. It is highly advisable to set this
        correctly in order to avoid filter artifacts in your evaluated frequency range.
    hset_info : tuple, list or :py:class:˚numpy.ndarray˚
        Contains information about the range of the up/downsampling factors.
        This should be a tuple, list or :py:class:˚numpy.ndarray˚ of (min, max, step).
    hset_accuracy : int
        floating point accuracy for the up/downsampling factor of the signal (default=4).

    Returns
    -------
    freqs : :py:class:`numpy.ndarray`
        The Frequencys associated with the (a-)periodic spectra.
    aperiodic : :py:class:`numpy.ndarray`
        The aperiodic component of the data.
    periodic : :py:class:`numpy.ndarray`
        The periodic component of the data.

    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
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

    This function can be used to seperate aperiodic from periodic power spectra
    using the IRASA algorithm (Wen & Liu, 2016) in a time resolved manner.

    Parameters
    ----------
    data : :py:class:˚numpy.ndarray˚
        The timeseries data used to extract aperiodic and periodic power spectra.
    fs : int
        The sampling frequency of the data.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract (a-)periodic spectra.
    freq_res : float
        The desired frequency resolution in Hz.
    smooth : bool
        Whether or not to smooth the time-frequency data before computing IRASA
        by averaging over adjacent fft bins using n_avgs.
    n_avgs :  int
        Number indicating the amount of fft bins to average across.
    win_duration : float
        The time width of window in seconds used to calculate the stffts.
    hop : int
        Time increment in signal samples for sliding window.
    win_duration : float
        The time width of window in seconds used to calculate the stffts.
    win_func : :py:class:`scipy.signal.windows`
        The desired window function. Can be any window function
        specified in :py:class:`scipy.signal.windows`.
        The default is `scipy.signal.windows.hann`.
    win_func_kwargs: dict
        A dictionary containing keyword arguments passed to win_func.
    dpss_settings:
        In case that you want to do multitapering using dpss
        we added a "sensible" preconfiguration as `scipy.signal.windows.dpss`
        requires more parameters than the window functions in :py:class:`scipy.signal.windows`.
        To change the settings adjust the parameter `win_func_kwargs`.
    filter_settings : tuple
        A tuple containing the cut-off of the High- and Lowpass filter.
        It is highly advisable to set this correctly in order to avoid
        filter artifacts in your evaluated frequency range.
    hset_info : tuple, list or :py:class:˚numpy.ndarray˚
        Contains information about the range of the up/downsampling factors.
        This should be a tuple, list or :py:class:˚numpy.ndarray˚ of (min, max, step).
    hset_accuracy : int
        floating point accuracy for the up/downsampling factor of the signal (default=4).


    Returns
    -------
    aperiodic : :py:class:`numpy.ndarray`
        The aperiodic component of the data.
    periodic : :py:class:`numpy.ndarray`
        The periodic component of the data.
    freqs : :py:class:`numpy.ndarray`
        The Frequencys associated with the (a-)periodic spectra.
    time : :py:class:`numpy.ndarray`
        The time bins in seconds associated with the (a-)periodic spectra.


    References
    ----------
        [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26.https://doi.org/10.1007/s10548-015-0448-0

    """

    # set parameters
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

    # NOTE: we need to transpose the data as crop_data extracts stuff from the last axis
    freq, sgramm_aperiodic, sgramm_periodic = _crop_data(band, freq, sgramm_aperiodic, sgramm_periodic, axis=0)

    # adjust time info (i.e. cut the padded stuff)
    tmax = data.shape[1] / fs
    t_mask = np.logical_and(time >= 0, time < tmax)

    return sgramm_aperiodic[:, t_mask], sgramm_periodic[:, t_mask], freq, time[t_mask]
