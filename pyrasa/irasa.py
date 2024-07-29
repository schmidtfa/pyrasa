import fractions
from collections.abc import Callable
from typing import TypedDict

import numpy as np
import scipy.signal as dsp
from scipy.signal import ShortTimeFFT

# from scipy.stats.mstats import gmean
from pyrasa.utils.irasa_utils import _check_irasa_settings, _crop_data, _find_nearest, _gen_time_from_sft, _get_windows


# TODO: Port to Cython
def _gen_irasa(
    data: np.ndarray,
    orig_spectrum: np.ndarray,
    fs: int,
    irasa_fun: Callable,
    hset: np.ndarray,
    irasa_kwargs: dict,
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
        irasa_kwargs['h'] = h
        irasa_kwargs['time_orig'] = time
        irasa_kwargs['up_down'] = 'up'
        spectrum_up = irasa_fun(data_up, int(fs * h), spectrum_only=True, **irasa_kwargs)
        irasa_kwargs['up_down'] = 'down'
        spectrum_dw = irasa_fun(data_down, int(fs / h), spectrum_only=True, **irasa_kwargs)

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
    irasa_kwargs: dict,
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
    irasa_kwargs : dict
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

    # Calculate original spectrum
    def _compute_psd_welch(
        data: np.ndarray,
        fs: int,
        window: str = 'hann',
        nperseg: int | None = None,
        noverlap: int | None = None,
        nfft: int | None = None,
        detrend: str = 'constant',
        return_onesided: bool = True,
        scaling: str = 'density',
        axis: int = -1,
        average: str = 'mean',
        spectrum_only: bool = False,
        h: float | None = None,
        time_orig: np.ndarray | None = None,
        up_down: str | None = None,
        # **irasa_kwargs ,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Function to compute power spectral densities using welchs method"""

        freq, psd = dsp.welch(
            data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            average=average,
        )

        if spectrum_only:
            return psd
        else:
            return freq, psd

    freq, psd = _compute_psd_welch(data, fs=fs, **irasa_kwargs)

    psd, psd_aperiodic, psd_periodic = _gen_irasa(
        data=np.squeeze(data),
        orig_spectrum=psd,
        fs=fs,
        irasa_fun=_compute_psd_welch,
        hset=hset,
        irasa_kwargs=irasa_kwargs,
    )

    freq, psd_aperiodic, psd_periodic = _crop_data(band, freq, psd_aperiodic, psd_periodic, axis=-1)

    return freq, psd_aperiodic, psd_periodic


# irasa sprint
def irasa_sprint(  # noqa PLR0915 C901
    data: np.ndarray,
    fs: int,
    band: tuple[float, float] = (1.0, 100.0),
    freq_res: float = 0.5,
    # smooth: bool = True,
    # n_avgs: list = [1],
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

    class IrasaKwargsTyped(TypedDict):
        mfft: int
        hop: int
        win_duration: float
        h: int | None
        up_down: str | None
        dpss_settings: dict
        win_kwargs: dict
        time_orig: None | np.ndarray
        # smooth: bool
        # n_avgs: list

    irasa_kwargs: IrasaKwargsTyped = {
        'mfft': mfft,
        'hop': hop,
        'win_duration': win_duration,
        'h': None,
        'up_down': None,
        'dpss_settings': dpss_settings,
        'win_kwargs': win_kwargs,
        'time_orig': None,
        #'smooth': smooth,
        #'n_avgs': n_avgs,
    }

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
        # smooth: bool = True,
        # n_avgs: list = [3],
        spectrum_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """Function to compute spectrograms"""

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

        # TODO: smoothing doesnt work properly
        # if smooth:
        #     avgs = []
        # def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
        #     return np.convolve(x, np.ones(w), 'valid') / w

        # def sgramm_smoother(sgramm: np.ndarray, n_avgs: int) -> np.ndarray:
        #     return np.array([_moving_average(sgramm[freq, :], w=n_avgs) for freq in range(sgramm.shape[0])])
        #     n_avgs_r = n_avgs[::-1]
        #     for avg, avg_r in zip(n_avgs, n_avgs_r):
        #         sgramm_fwd = sgramm_smoother(sgramm=np.squeeze(sgramm), n_avgs=avg)[:, avg_r:]
        #         sgramm_bwd = sgramm_smoother(sgramm=np.squeeze(sgramm)[:, ::-1], n_avgs=avg)[:, ::-1][:, avg_r:]
        #         sgramm_n = gmean([sgramm_fwd, sgramm_bwd], axis=0)
        #         avgs.append(sgramm_n)

        #     sgramm = np.median(avgs, axis=0)
        #     sgramm = sgramm[np.newaxis, :, :]

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

        if spectrum_only:
            return sgramm
        else:
            return freq, time, sgramm

    # get time and frequency info
    freq, time, sgramm = _compute_sgramm(data, fs, **irasa_kwargs)

    sgramm, sgramm_aperiodic, sgramm_periodic = _gen_irasa(
        data=data,
        orig_spectrum=sgramm,
        fs=fs,
        irasa_fun=_compute_sgramm,
        hset=hset,
        irasa_kwargs=dict(irasa_kwargs),
        time=time,
    )

    # NOTE: we need to transpose the data as crop_data extracts stuff from the last axis
    freq, sgramm_aperiodic, sgramm_periodic = _crop_data(band, freq, sgramm_aperiodic, sgramm_periodic, axis=0)

    # adjust time info (i.e. cut the padded stuff)
    tmax = data.shape[1] / fs
    t_mask = np.logical_and(time >= 0, time < tmax)

    return sgramm_aperiodic[:, t_mask], sgramm_periodic[:, t_mask], freq, time[t_mask]
