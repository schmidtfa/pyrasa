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
    duration : float, required
        The duration (in seconds) of each segment used to calculate the power spectral density (PSD).
        This must be less than the total duration of the data.
    overlap : float or int, optional, default: 50
        The overlap between segments, specified as a percentage (0-100).
    hset_info : tuple of (float, float, float), optional, default: (1.05, 2.0, 0.05)
        Contains the range of up/downsampling factors used in the IRASA algorithm.
        This should be a tuple specifying the (min, max, step) values for the resampling.

    Returns
    -------
    IrasaRaw
        A custom object containing the separated aperiodic and periodic components of the data:
        - `periodic`: An instance of `PeriodicSpectrumArray`, which includes the periodic
          (oscillatory) component of the power spectrum.
        - `aperiodic`: An instance of `AperiodicSpectrumArray`, which includes the aperiodic
          (fractal) component of the power spectrum.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum
    of Neurophysiological Signal. Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

    Notes
    -----
    - Ensure that `data` does not contain any bad channels (`data.info['bads']` should be empty),
      as this could affect the results.
    - The overlap percentage should be carefully chosen to balance between segment independence
      and sufficient data for analysis. A value between 0 and 100% is valid.
    - The function will raise assertions if the input parameters are not consistent with the
      expected formats or if the provided `duration` exceeds the length of the data.

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
    Separate aperiodic from periodic power spectra using the IRASA algorithm for Epochs data.

    This function applies the Irregular Resampling Auto-Spectral Analysis (IRASA) algorithm
    as described by Wen & Liu (2016) to decompose the power spectrum of neurophysiological
    signals into aperiodic (fractal) and periodic (oscillatory) components. It is specifically
    designed for time-series data in `mne.Epochs` format, making it suitable for event-related
    EEG/MEG analyses.

    Parameters
    ----------
    data : mne.Epochs
        The time-series data used to extract aperiodic and periodic power spectra.
        This should be an instance of `mne.Epochs`.
    band : tuple of (float, float), optional, default: (1.0, 100.0)
        A tuple specifying the lower and upper bounds of the frequency range (in Hz) used
        for extracting the aperiodic and periodic spectra.
    hset_info : tuple of (float, float, float), optional, default: (1.05, 2.0, 0.05)
        Contains the range of up/downsampling factors used in the IRASA algorithm.
        This should be a tuple specifying the (min, max, step) values for the resampling.

    Returns
    -------
    aperiodic : AperiodicEpochsSpectrum
        The aperiodic component of the data as an `AperiodicEpochsSpectrum` object.
    periodic : PeriodicEpochsSpectrum
        The periodic component of the data as a `PeriodicEpochsSpectrum` object.

    References
    ----------
    Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory Components in the Power Spectrum
    of Neurophysiological Signal. Brain Topography, 29(1), 13–26.
    https://doi.org/10.1007/s10548-015-0448-0

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
