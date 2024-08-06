import mne
import numpy as np

from pyrasa.irasa import irasa
from pyrasa.irasa_mne.mne_objs import (
    AperiodicEpochsSpectrum,
    AperiodicSpectrumArray,
    IrasaEpoched,
    IrasaRaw,
    PeriodicEpochsSpectrum,
    PeriodicSpectrumArray,
)


def irasa_raw(
    data: mne.io.Raw,
    band: tuple[float, float] = (1.0, 100.0),
    duration: float | None = None,
    overlap: float | int = 50,
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
) -> IrasaRaw:
    """
    This function can be used to seperate aperiodic from periodic power spectra using
    the IRASA algorithm (Wen & Liu, 2016).

    Parameters
    ----------
    data : :py:class:˚mne.io.BaseRaw˚
        The timeseries data used to extract aperiodic and periodic power spectra.
        Should be :py:class:˚mne.io.BaseRaw˚ in which case 'fs' and 'filter_settings'
        will be automatically extracted.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract
        (a-)periodic spectra.
    duration : float
        The time window of each segment in seconds used to calculate the psd.
    overlap : float
        The overlap between segments in percent
    hset_info : tuple, list or :py:class:˚numpy.ndarray˚
        Contains information about the range of the up/downsampling factors.
        This should be a tuple, list or :py:class:˚numpy.ndarray˚ of (min, max, step).
    as_array : bool
        The function returns an :py:class:˚mne.time_frequency.SpectrumArray˚ if set to False (default)
        if set to True the data is returned as :py:class:`numpy.ndarray`.

    Returns
    -------
        aperiodic : :py:class:˚mne.time_frequency.SpectrumArray˚
            The aperiodic component of the data as an mne.time_frequency.SpectrumArray object.
        periodic : :py:class:˚mne.time_frequency.SpectrumArray˚
            The periodic component of the data as mne.time_frequency.SpectrumArray object.


    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26.
        https://doi.org/10.1007/s10548-015-0448-0

    """

    # set parameters & safety checks
    # ensure that input data is in the right format
    assert isinstance(data, mne.io.BaseRaw), 'Data should be of type mne.BaseRaw'
    assert data.info['bads'] == [], (
        'Data should not contain bad channels ' 'as this might mess up the creation of the returned data structure'
    )

    # extract relevant info from mne object
    info = data.info.copy()
    fs = int(data.info['sfreq'])
    data_array = data.get_data()

    overlap /= 100
    # assert isinstance(duration, int | float), 'You need to set the duration of your time window in seconds'
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

    irasa_spectrum = irasa(
        data_array,
        fs=fs,
        band=band,
        filter_settings=(data.info['highpass'], data.info['lowpass']),
        hset_info=hset_info,
        psd_kwargs=kwargs_psd,
    )

    return IrasaRaw(
        periodic=PeriodicSpectrumArray(irasa_spectrum.periodic, info, freqs=irasa_spectrum.freqs),
        aperiodic=AperiodicSpectrumArray(irasa_spectrum.aperiodic, info, freqs=irasa_spectrum.freqs),
    )


def irasa_epochs(
    data: mne.Epochs,
    band: tuple[float, float] = (1.0, 100.0),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
) -> IrasaEpoched:
    """
    This function can be used to seperate aperiodic from periodic power spectra
    using the IRASA algorithm (Wen & Liu, 2016).

    Parameters
    ----------
    data : :py:class:˚mne.io.BaseEpochs˚
        The timeseries data used to extract aperiodic and periodic power spectra.
        Should be :py:class:˚mne.io.BaseEpochs˚ in which case 'fs', 'filter_settings',
        'duration' and 'overlap' will be automatically extracted.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract
        (a-)periodic spectra.
    hset_info : tuple, list or :py:class:˚numpy.ndarray˚
        Contains information about the range of the up/downsampling factors.
        This should be a tuple, list or :py:class:˚numpy.ndarray˚ of (min, max, step).
    as_array : bool
        The function returns an :py:class:˚mne.time_frequency.EpochsSpectrumArray˚ if set to False (default)
        if set to True the data is returned as :py:class:`numpy.ndarray`.

    Returns
    -------
        aperiodic : :py:class:˚mne.time_frequency.EpochsSpectrumArray˚
            The aperiodic component of the data as an mne.time_frequency.EpochsSpectrumArray object.
        periodic : :py:class:˚mne.time_frequency.EpochsSpectrumArray˚
            The periodic component of the data as mne.time_frequency.EpochsSpectrumArray object.


    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26.
        https://doi.org/10.1007/s10548-015-0448-0

    """

    # set parameters & safety checks
    # ensure that input data is in the right format
    assert isinstance(data, mne.BaseEpochs), 'Data should be of type mne.BaseEpochs'
    assert (
        data.info['bads'] == []
    ), 'Data should not contain bad channels as this might mess up the creation of the returned data structure'

    info = data.info.copy()
    fs = int(data.info['sfreq'])
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
        irasa_spectrum = irasa(
            epoch,
            fs=fs,
            band=band,
            filter_settings=(data.info['highpass'], data.info['lowpass']),
            hset_info=hset_info,
            psd_kwargs=kwargs_psd,
        )
        psd_list_aperiodic.append(irasa_spectrum.aperiodic.copy())
        psd_list_periodic.append(irasa_spectrum.periodic.copy())

    psds_aperiodic = np.array(psd_list_aperiodic)
    psds_periodic = np.array(psd_list_periodic)

    return IrasaEpoched(
        periodic=PeriodicEpochsSpectrum(
            psds_periodic, info, freqs=irasa_spectrum.freqs, events=events, event_id=event_ids
        ),
        aperiodic=AperiodicEpochsSpectrum(
            psds_aperiodic, info, freqs=irasa_spectrum.freqs, events=events, event_id=event_ids
        ),
    )
