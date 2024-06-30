import numpy as np
import mne
from pyrasa.irasa_mne.mne_objs import (PeriodicSpectrumArray, AperiodicSpectrumArray,
                                       PeriodicEpochsSpectrum, AperiodicEpochsSpectrum)
import mne
from pyrasa.irasa import irasa
from pyrasa.utils.irasa_utils import _check_input_data_mne

def irasa_raw(data,                  
                fs=None,
                band=(1,100),
                duration=None, 
                overlap=50, 
                hset_info=(1., 2., 0.05),
                as_array=False):

    '''
    This function can be used to seperate aperiodic from periodic power spectra using 
    the IRASA algorithm (Wen & Liu, 2016). This implementation of the IRASA algorithm 
    is based on the yasa.irasa function in (Vallat & Walker, 2021).

    Parameters
    ----------
    data : :py:class:˚numpy.ndarray˚ or :py:class:˚mne.io.BaseRaw˚ 
        The timeseries data used to extract aperiodic and periodic power spectra. 
        Can also be :py:class:˚mne.io.BaseRaw˚ in which case 'fs' and 'filter_settings' 
        will be automatically extracted.
    fs : int
        The sampling frequency of the data. Can be omitted if data is :py:class:˚mne.io.BaseRaw˚.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract 
        (a-)periodic spectra.
    filter_settings : tuple
        A tuple containing the cut-off of the High- and Lowpass filter. 
        It is highly advisable to set this correctly in order to avoid filter artifacts 
        in your evaluated frequency range. Can be omitted if data is :py:class:˚mne.io.BaseRaw˚.
    duration : float
        The time window of each segment in seconds used to calculate the psd.
    overlap : float
        The overlap between segments in percent
    return_type : 
        The returned datatype, when using :py:class:˚mne.io.BaseRaw˚ can be either
        ˚mne.time_frequency.SpectrumArray˚ or :py:class:`numpy.ndarray`. When the input data is of type
        :py:class:`numpy.ndarray` the returned data type is always :py:class:`numpy.ndarray`.

    Returns
    -------
    if data : :py:class:`numpy.ndarray`
        freqs : :py:class:`numpy.ndarray`
            The Frequencys associated with the (a-)periodic spectra.
        aperiodic : :py:class:`numpy.ndarray`
            The aperiodic component of the data.
        periodic : :py:class:`numpy.ndarray`
            The periodic component of the data.

    elif data : :py:class:˚mne.io.BaseRaw˚
        aperiodic : :py:class:˚mne.time_frequency.SpectrumArray˚
            The aperiodic component of the data as an mne.time_frequency.SpectrumArray object.
        periodic : :py:class:˚mne.time_frequency.SpectrumArray˚
            The periodic component of the data as anmne.time_frequency.SpectrumArray object.

            
    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26.
        https://doi.org/10.1007/s10548-015-0448-0
    [2] Vallat, Raphael, and Matthew P. Walker. “An open-source, 
        high-performance tool for automated sleep staging.” 
        Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

    '''

    #set parameters & safety checks
    #ensure that input data is in the right format
    assert isinstance(data, mne.io.BaseRaw), 'Data should be of type mne.BaseRaw'
    assert data.info['bads'] == [], ('Data should not contain bad channels '
                                     'as this might mess up the creation of the returned data structure')
    
    #extract relevant info from mne object
    info = data.info.copy()
    fs = data.info["sfreq"] 
    data_array = data.get_data() 

    overlap /= 100
    assert isinstance(duration, (int, float)), 'You need to set the duration of your time window in seconds'
    assert data_array.shape[1] > int(fs*duration), 'The duration for each segment cant be longer than the actual data'
    assert np.logical_and(overlap < 1,  overlap > 0), 'The overlap between segments cant be larger than 100% or less than 0%'

    _check_input_data_mne(data, hset_info, band)

    nfft = 2**(np.ceil(np.log2(int(fs*duration*np.max(hset_info)))))
    kwargs_psd = {'window': 'hann',
                  'average': 'median',
                  'nfft': nfft,
                  'nperseg': int(fs*duration), 
                  'noverlap': int(fs*duration*overlap)}
    
    freq, psd_aperiodic, psd_periodic = irasa(data_array, fs=fs, band=band, 
                                              hset_info=hset_info, kwargs_psd=kwargs_psd)
    
    if as_array==True:
        return psd_aperiodic, psd_periodic, freq
    
    else:
        aperiodic = AperiodicSpectrumArray(psd_aperiodic, info, freqs=freq)
        periodic = PeriodicSpectrumArray(psd_periodic, info, freqs=freq)

        return aperiodic, periodic


def irasa_epochs(data,                  
          band=(1,100),
          hset_info=(1., 2., 0.05),
          as_array=False):

    '''
    This function can be used to seperate aperiodic from periodic power spectra 
    using the IRASA algorithm (Wen & Liu, 2016). This implementation of the IRASA algorithm
    is based on the yasa.irasa function in (Vallat & Walker, 2021).

    Parameters
    ----------
    data : :py:class:˚mne.io.BaseEpochs˚
        The timeseries data used to extract aperiodic and periodic power spectra. 
        Should be :py:class:˚mne.io.BaseEpochs˚ in which case 'fs', 'filter_settings', 
        'duration' and 'overlap' will be automatically extracted.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract 
        (a-)periodic spectra.
    return_type : :py:class: ˚numpy.ndarray˚ or :py:class:˚mne.io.BaseEpochs˚
        The returned datatype, when using :py:class:˚mne.io.BaseEpochs˚ can be either
        ˚mne.time_frequency.EpochsSpectrumArray˚ or :py:class:`numpy.ndarray`.
        When the input data is of type.

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
    [2] Vallat, Raphael, and Matthew P. Walker. “An open-source, 
        high-performance tool for automated sleep staging.” 
        Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

    '''

    #set parameters & safety checks
    #ensure that input data is in the right format
    assert isinstance(data, mne.BaseEpochs),  'Data should be of type mne.BaseEpochs'
    assert data.info['bads'] == [], 'Data should not contain bad channels as this might mess up the creation of the returned data structure'
    assert isinstance(hset_info, tuple), 'hset should be a tuple of (min, max, step)'

    info = data.info.copy()
    fs = data.info["sfreq"]
    events = data.events
    event_ids = data.event_id

    data_array = data.get_data(copy=True)

    _check_input_data_mne(data, hset_info, band)
    
    #TODO: check if hset.max() is really max
    nfft = 2**(np.ceil(np.log2(int(data_array.shape[2]*np.max(hset_info)))))

    #TODO: does zero padding make sense?
    kwargs_psd = {'window': 'hann',
                 #'nperseg': data_array.shape[2],
                  'nfft': nfft,
                  'noverlap': 0,}

    # Do the actual IRASA stuff..
    psd_list_aperiodic, psd_list_periodic = [], []
    for epoch in data_array:
    
        freq, psd_aperiodic, psd_periodic = irasa(epoch, fs=fs, band=band, 
                                                  hset_info=hset_info, kwargs_psd=kwargs_psd)
        psd_list_aperiodic.append(psd_aperiodic)
        psd_list_periodic.append(psd_periodic)
    
    psd_aperiodic = np.array(psd_list_aperiodic)
    psd_periodic = np.array(psd_list_periodic)

    if as_array==True:
        return psd_aperiodic, psd_periodic, freq
    else:
        aperiodic = AperiodicEpochsSpectrum(psd_aperiodic, info, freqs=freq, events=events, event_id=event_ids)
        periodic = PeriodicEpochsSpectrum(psd_periodic, info, freqs=freq, events=events, event_id=event_ids)

        return aperiodic, periodic
