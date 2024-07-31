"""Interface to use the IRASA algorithm with MNE objects."""

import mne
import numpy as np

from pyrasa.irasa import irasa
from pyrasa.irasa_mne.mne_objs import (
    AperiodicEpochsSpectrum,
    AperiodicSpectrumArray,
    PeriodicEpochsSpectrum,
    PeriodicSpectrumArray,
)


def irasa_raw(
    data: mne.io.Raw,
    band: tuple[float, float] = (1.0, 100.0),
    duration: float | None = None,
    overlap: float | int = 50,
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    as_array: bool = False,
) -> tuple[AperiodicSpectrumArray, PeriodicSpectrumArray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate aperiodic from periodic power spectra using the IRASA algorithm.

    Parameters
    ----------
    data : mne.io.Raw
        The timeseries data used to extract aperiodic and periodic power spectra.
    band : tuple of float, optional
        The lower and upper band of the frequency range used to extract (a-)periodic spectra.
        Defaults to (1.0, 100.0).
    duration : float, optional
        The time window of each segment in seconds used to calculate the PSD.
        If None, the entire duration of the data is used.
    overlap : float or int, optional
        The overlap between segments in percent. Defaults to 50.
    hset_info : tuple of float, optional
        Information about the range of the up/downsampling factors. Should be a tuple of (min, max, step).
        Defaults to (1.05, 2.0, 0.05).
    as_array : bool, optional
        If True, the data is returned as numpy.ndarray. If False, returns mne.time_frequency.SpectrumArray.
        Defaults to False.

    Returns
    -------
    tuple of AperiodicSpectrumArray and PeriodicSpectrumArray or tuple of numpy.ndarray
        The aperiodic and periodic components of the data.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
    Components in the Power Spectrum of Neurophysiological Signal.
    Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

    """
    # Set parameters & safety checks
    # Ensure that input data is in the right format
    assert isinstance(data, mne.io.BaseRaw), 'Data should be of type mne.io.BaseRaw'
    assert (
        data.info['bads'] == []
    ), 'Data should not contain bad channels as this might mess up the creation of the returned data structure'

    # Extract relevant info from mne object
    info = data.info.copy()
    fs = data.info['sfreq']
    data_array = data.get_data()

    overlap /= 100
    assert isinstance(duration, int | float), 'You need to set the duration of your time window in seconds'
    assert data_array.shape[1] > int(fs * duration), 'The duration for each segment cant be longer than the actual data'
    assert np.logical_and(
        overlap < 1, overlap > 0
    ), 'The overlap between segments cant be larger than 100% or less than 0%'

    nfft = 2 ** (np.ceil(np.log2(int(fs * duration * np.max(hset_info)))))
    kwargs_psd = {
        'window': 'hann',
        'average': 'median',
        'nfft': nfft,
        'nperseg': int(fs * duration),
        'noverlap': int(fs * duration * overlap),
    }

    freq, psd_aperiodic, psd_periodic = irasa(
        data_array,
        fs=fs,
        band=band,
        filter_settings=(data.info['highpass'], data.info['lowpass']),
        hset_info=hset_info,
        psd_kwargs=kwargs_psd,
    )

    if as_array:
        return psd_aperiodic, psd_periodic, freq
    else:
        aperiodic = AperiodicSpectrumArray(psd_aperiodic, info, freqs=freq)
        periodic = PeriodicSpectrumArray(psd_periodic, info, freqs=freq)

        return aperiodic, periodic


def irasa_epochs(
    data: mne.Epochs,
    band: tuple[float, float] = (1.0, 100.0),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    as_array: bool = False,
) -> tuple[AperiodicSpectrumArray, PeriodicSpectrumArray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate aperiodic from periodic power spectra using the IRASA algorithm.

    Parameters
    ----------
    data : mne.Epochs
        The timeseries data used to extract aperiodic and periodic power spectra.
    band : tuple of float, optional
        The lower and upper band of the frequency range used to extract (a-)periodic spectra.
        Defaults to (1.0, 100.0).
    hset_info : tuple of float, optional
        Information about the range of the up/downsampling factors. Should be a tuple of (min, max, step).
        Defaults to (1.05, 2.0, 0.05).
    as_array : bool, optional
        If True, the data is returned as numpy.ndarray. If False, returns mne.time_frequency.EpochsSpectrumArray.
        Defaults to False.

    Returns
    -------
    tuple of AperiodicSpectrumArray and PeriodicSpectrumArray or tuple of numpy.ndarray
        The aperiodic and periodic components of the data.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
    Components in the Power Spectrum of Neurophysiological Signal.
    Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

    """
    # Set parameters & safety checks
    # Ensure that input data is in the right format
    assert isinstance(data, mne.BaseEpochs), 'Data should be of type mne.BaseEpochs'
    assert (
        data.info['bads'] == []
    ), 'Data should not contain bad channels as this might mess up the creation of the returned data structure'

    info = data.info.copy()
    fs = data.info['sfreq']
    events = data.events
    event_ids = data.event_id

    data_array = data.get_data(copy=True)

    # TODO: check if hset.max() is really max
    nfft = 2 ** (np.ceil(np.log2(int(data_array.shape[2] * np.max(hset_info)))))

    # TODO: does zero padding make sense?
    kwargs_psd = {
        'window': 'hann',
        'nperseg': None,
        'nfft': nfft,
        'noverlap': 0,
    }

    # Do the actual IRASA stuff..
    psd_list_aperiodic, psd_list_periodic = [], []
    for epoch in data_array:
        freq, psd_aperiodic, psd_periodic = irasa(
            epoch,
            fs=fs,
            band=band,
            filter_settings=(data.info['highpass'], data.info['lowpass']),
            hset_info=hset_info,
            psd_kwargs=kwargs_psd,
        )
        psd_list_aperiodic.append(psd_aperiodic)
        psd_list_periodic.append(psd_periodic)

    psd_aperiodic = np.array(psd_list_aperiodic)
    psd_periodic = np.array(psd_list_periodic)

    if as_array:
        return psd_aperiodic, psd_periodic, freq
    else:
        aperiodic = AperiodicEpochsSpectrum(psd_aperiodic, info, freqs=freq, events=events, event_id=event_ids)
        periodic = PeriodicEpochsSpectrum(psd_periodic, info, freqs=freq, events=events, event_id=event_ids)

        return aperiodic, periodic
