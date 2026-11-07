"""Interface to use the IRASA algorithm with MNE objects."""

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
    Separate aperiodic from periodic power spectra using the IRASA algorithm.

    This function applies the Irregular Resampling Auto-Spectral Analysis (IRASA) algorithm
    as described by Wen & Liu (2016) to decompose the power spectrum of neurophysiological
    signals into aperiodic (fractal) and periodic (oscillatory) components. This function is
    essentially a wrapper function for `pyrasa.irasa`

    Parameters
    ----------
    data : mne.io.Raw
        The time-series data from which the aperiodic and periodic power spectra are extracted.
        This should be an instance of `mne.io.Raw`. The function will automatically extract
        relevant parameters such as sampling frequency (`sfreq`) and filtering settings from the `mne` object
        to make sure the model is specified correctly.
    band : tuple of (float, float), optional, default: (1.0, 100.0)
        A tuple specifying the lower and upper bounds of the frequency range (in Hz) used
        for extracting the aperiodic and periodic spectra.
    duration : float | None, default None
        Duration of each segment used to estimate the power spectral
        density (PSD), in seconds. Must be provided and must be shorter
        than the total recording duration.
    overlap : float | int, default 50
        Overlap between consecutive segments, expressed as a percentage.
        Must be strictly between 0 and 100.
    hset_info : tuple of (float, float, float), optional, default: (1.05, 2.0, 0.05)
        Contains the range of up/downsampling factors used in the IRASA algorithm.
        This should be a tuple specifying the (min, max, step) values for the resampling.

    Returns
    -------
    IrasaRaw
        Object containing the IRASA decomposition with two MNE-compatible
        spectral representations:

        - `periodic`: PeriodicSpectrumArray containing the periodic
        (oscillatory) component of the power spectrum.
        - `aperiodic`: AperiodicSpectrumArray containing the aperiodic
        (fractal) component of the power spectrum.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum
    of Neurophysiological Signal. Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

    Notes
    -----
    The input data must not contain channels marked as bad in
    `data.info['bads']`.

    Power spectral densities are estimated using overlapping segments
    and median averaging. The recording's high-pass and low-pass filter
    settings are passed to the underlying IRASA implementation.
    """

    # set parameters & safety checks
    # ensure that input data is in the right format
    assert isinstance(data, mne.io.BaseRaw), 'Data should be of type mne.BaseRaw'
    assert data.info['bads'] == [], (
        'Data should not contain bad channels ' 'as this might mess up the creation of the returned data structure'
    )

    # extract relevant info from mne object
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
        **kwargs_psd,
    )

    return IrasaRaw(
        periodic=PeriodicSpectrumArray(
            irasa_spectrum.periodic, info, freqs=irasa_spectrum.freqs, aperiodic_spectrum=irasa_spectrum.aperiodic
        ),
        aperiodic=AperiodicSpectrumArray(irasa_spectrum.aperiodic, info, freqs=irasa_spectrum.freqs),
    )


def irasa_epochs(
    data: mne.Epochs,
    band: tuple[float, float] = (1.0, 100.0),
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
) -> IrasaEpoched:
    """
    Separate aperiodic from periodic power spectra using the IRASA algorithm for Epochs data.

    This function applies the Irregular Resampling Auto-Spectral Analysis (IRASA) algorithm
    as described by Wen & Liu (2016) to decompose the power spectrum of neurophysiological
    signals into aperiodic (fractal) and periodic (oscillatory) components. It is specifically
    designed for time-series data in `mne.Epochs` format, making it suitable for event-related
    EEG/MEG analyses.

    Parameters
    ----------
    data : mne.Epochs
        Epoched neurophysiological time-series data used to estimate
        the periodic and aperiodic power spectra. Any MNE object
        inheriting from `mne.BaseEpochs` is supported.
    band : tuple of (float, float), optional, default: (1.0, 100.0)
        A tuple specifying the lower and upper bounds of the frequency range (in Hz) used
        for extracting the aperiodic and periodic spectra.
    hset_info : tuple of (float, float, float), optional, default: (1.05, 2.0, 0.05)
        Contains the range of up/downsampling factors used in the IRASA algorithm.
        This should be a tuple specifying the (min, max, step) values for the resampling.

    Returns
    -------
    IrasaEpoched
    Object containing the IRASA decomposition for each epoch and
    channel, with two MNE-compatible spectral representations:

    - `periodic`: PeriodicEpochsSpectrum containing the periodic
      (oscillatory) component of the power spectrum.
    - `aperiodic`: AperiodicEpochsSpectrum containing the aperiodic
      (fractal) component of the power spectrum.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum
    of Neurophysiological Signal. Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

    Notes
    -----
    IRASA is applied independently to each epoch using its full duration
    for spectral estimation. No averaging across epochs is performed.

    The original event information and event ID mapping are preserved
    in the returned spectral objects.

    The input data must not contain channels marked as bad in
    `data.info['bads']`.

    """

    # set parameters & safety checks
    # ensure that input data is in the right format
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
            **kwargs_psd,
        )
        psd_list_aperiodic.append(irasa_spectrum.aperiodic.copy())
        psd_list_periodic.append(irasa_spectrum.periodic.copy())

    psds_aperiodic = np.array(psd_list_aperiodic)
    psds_periodic = np.array(psd_list_periodic)

    return IrasaEpoched(
        periodic=PeriodicEpochsSpectrum(
            psds_periodic,
            info,
            freqs=irasa_spectrum.freqs,
            events=events,
            event_id=event_ids,
        ),
        aperiodic=AperiodicEpochsSpectrum(
            psds_aperiodic, info, freqs=irasa_spectrum.freqs, events=events, event_id=event_ids
        ),
    )
