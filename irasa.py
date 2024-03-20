import scipy.signal as dsp
from scipy.signal import ShortTimeFFT
import fractions
import numpy as np
import mne


from irasa_utils import (_crop_data, _gen_time_from_sft, _find_nearest,
                          _check_input_data, _check_psd_settings)
#TODO: Port to Cython

#%% irasa
def irasa(data, fs, band, kwargs_psd, hset_info=(1.1, 1.95, 0.05)):

    '''
    This function can be used to seperate aperiodic from periodic power spectra using the IRASA algorithm (Wen & Liu, 2016).
    This implementation of the IRASA algorithm is based on the yasa.irasa function in (Vallat & Walker, 2021).

    WARNING: This is the raw irasa algorithm that gives you maximal control over all parameters. 
    It will not inform or warn you when your parameter settings are invalid. Use this only if you know what you do!
    Otherwise it is recommend to use the irasa_raw, irasa_epochs or irasa_sprint functions.

    Parameters
    ----------
    data : :py:class:˚numpy.ndarray˚
        The timeseries data used to extract aperiodic and periodic power spectra. 
    fs : int
        The sampling frequency of the data. Can be omitted if data is :py:class:˚mne.io.BaseRaw˚.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract (a-)periodic spectra.
    filter_settings : tuple
        A tuple containing the cut-off of the High- and Lowpass filter. 
        It is highly advisable to set this correctly in order to avoid filter artifacts in your evaluated frequency range.
    duration : float
        The time window of each segment in seconds used to calculate the psd.
    kwargs_psd : dict
        A dictionary containing all the keyword arguments that are passed onto 

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
    [2] Vallat, Raphael, and Matthew P. Walker. “An open-source, 
        high-performance tool for automated sleep staging.” 
        Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

    '''

    #Minimal safety checks
    assert isinstance(data, np.ndarray), 'Data should either be a numpy array or an mne python Raw object.'
    if data.ndim == 1:
        data = data[np.newaxis, :]
    assert data.ndim == 2, 'Data shape needs to be either of shape (Channels, Samples) or (Samples, ).'
    assert isinstance(hset_info, tuple), 'hset should be a tuple of (min, max, step)'

    hset = np.round(np.arange(*hset_info), 4)
    # Calculate original spectrum
    freq, psd = dsp.welch(data, fs=fs, **kwargs_psd)

    psds = np.zeros((len(hset), *psd.shape))
    for i, h in enumerate(hset):

        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator

        # Much faster than FFT-based resampling
        data_up = dsp.resample_poly(data, up, down, axis=-1)
        data_down = dsp.resample_poly(data, down, up, axis=-1)

        # Calculate an up/downsampled version of the PSD using same params as original
        _, psd_up = dsp.welch(data_up, fs * h,  **kwargs_psd)
        _, psd_dw = dsp.welch(data_down, fs / h, **kwargs_psd)

        # geometric mean between up and downsampled
        psds[i, :] = np.sqrt(psd_up * psd_dw)

    psd_aperiodic = np.median(psds, axis=0)
    psd_periodic = psd - psd_aperiodic

    freq, psd_aperiodic, psd_periodic = _crop_data(band, freq, psd_aperiodic, psd_periodic, axis=-1)

    return freq, psd_aperiodic, psd_periodic
    


def irasa_raw(data,                  
                fs=None,
                band=(1,100),
                duration=None, 
                overlap=50, 
                hset_info=(1.1, 1.95, 0.05),
                return_type=np.array):

    '''
    This function can be used to seperate aperiodic from periodic power spectra using the IRASA algorithm (Wen & Liu, 2016).
    This implementation of the IRASA algorithm is based on the yasa.irasa function in (Vallat & Walker, 2021).

    Parameters
    ----------
    data : :py:class:˚numpy.ndarray˚ or :py:class:˚mne.io.BaseRaw˚ 
        The timeseries data used to extract aperiodic and periodic power spectra. 
        Can also be :py:class:˚mne.io.BaseRaw˚ in which case 'fs' and 'filter_settings' will be automatically extracted.
    fs : int
        The sampling frequency of the data. Can be omitted if data is :py:class:˚mne.io.BaseRaw˚.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract (a-)periodic spectra.
    filter_settings : tuple
        A tuple containing the cut-off of the High- and Lowpass filter. 
        It is highly advisable to set this correctly in order to avoid filter artifacts in your evaluated frequency range.
        Can be omitted if data is :py:class:˚mne.io.BaseRaw˚.
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
    assert isinstance(data, mne.io.BaseRaw), 'Data should be of type mne.BaseEpochs'
    assert data.info['bads'] == [], 'Data should not contain bad channels as this might mess up the creation of the returned data structure'
    _check_input_data(data, hset_info, band)

    
    info = data.info.copy()
    fs = data.info["sfreq"] 
    data_array = data.get_data() 

    _check_psd_settings(data_array, fs, duration, overlap)

    nfft = 2**(np.ceil(np.log2(int(fs*duration*np.max(hset_info)))))
    kwargs_psd = {'window': 'hann',
                  'average': 'median',
                  'nfft': nfft,
                  'nperseg': int(fs*duration), 
                  'noverlap': int(fs*duration*overlap)}
    
    freq, psd_aperiodic, psd_periodic = irasa(data_array, fs=fs, band=band, hset=hset_info, kwargs_psd=kwargs_psd)
    
    if return_type == isinstance(np.array):
        return psd_aperiodic, psd_periodic, freq
    
    elif return_type == isinstance(mne.time_frequency.SpectrumArray):
        aperiodic = mne.time_frequency.SpectrumArray(psd_aperiodic, info, freqs=freq)
        periodic = mne.time_frequency.SpectrumArray(psd_periodic, info, freqs=freq)

        return aperiodic, periodic


def irasa_epochs(data,                  
          band=(1,100),
          hset_info=(1.1, 1.95, 0.05),
          return_type=np.array):

    '''
    This function can be used to seperate aperiodic from periodic power spectra using the IRASA algorithm (Wen & Liu, 2016).
    This implementation of the IRASA algorithm is based on the yasa.irasa function in (Vallat & Walker, 2021).

    Parameters
    ----------
    data : :py:class:˚mne.io.BaseEpochs˚
        The timeseries data used to extract aperiodic and periodic power spectra. 
        Should be :py:class:˚mne.io.BaseEpochs˚ in which case 'fs', 'filter_settings', 'duration' and 'overlap' will be automatically extracted.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract (a-)periodic spectra.
    return_type : :py:class: ˚numpy.ndarray˚ or :py:class:˚mne.io.BaseEpochs˚
        The returned datatype, when using :py:class:˚mne.io.BaseEpochs˚ can be either
        ˚mne.time_frequency.EpochsSpectrumArray˚ or :py:class:`numpy.ndarray`. When the input data is of type.

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

    _check_input_data(data, hset_info, band)

    info = data.info.copy()
    fs = data.info["sfreq"]
    events = data.events
    event_ids = data.event_id

    data_array = data.get_data()
    
    #TODO: check if hset.max() is really max
    nfft = 2**(np.ceil(np.log2(int(data_array.shape[2]*np.max(hset_info)))))

    #TODO: does zero padding make sense?
    kwargs_psd = {'window': 'hann',
                    'nperseg': data.shape[2],
                    'nfft': nfft,
                    'noverlap': 0,}

    # Do the actual IRASA stuff..
    psd_list_aperiodic, psd_list_periodic = [], []
    for epoch in data_array:
    
        freq, psd_aperiodic, psd_periodic = irasa(epoch, fs=fs, band=band, hset=hset_info,, kwargs_psd=kwargs_psd)
        psd_list_aperiodic.append(psd_aperiodic)
        psd_list_periodic.append(psd_periodic)
    
    psd_aperiodic = np.array(psd_list_aperiodic)
    psd_periodic = np.array(psd_list_periodic)

    if return_type == isinstance(np.array):
        return psd_aperiodic, psd_periodic, freq
    elif return_type == isinstance(mne.time_frequency.EpochsSpectrumArray):
        aperiodic = mne.time_frequency.EpochsSpectrumArray(psd_aperiodic, info, freqs=freq, events=events, event_id=event_ids)
        periodic = mne.time_frequency.EpochsSpectrumArray(psd_periodic, info, freqs=freq, events=events, event_id=event_ids)

        return aperiodic, periodic




#%% irasa sprint

def irasa_sprint(x,
                 fs,
                 filter_settings,
                 band=(1,100),
                 duration=4,
                 hset=(1.1, 1.95, 0.05)):

    '''

    This function can be used to seperate aperiodic from periodic power spectra using the IRASA algorithm (Wen & Liu, 2016)
    in a time resolved manner.
    
    Parameters
    ----------
    data : :py:class:˚numpy.ndarray˚, :py:class:˚mne.io.BaseRaw˚ or :py:class:˚mne.io.BaseEpochs˚
        The timeseries data used to extract aperiodic and periodic power spectra. 
        Can also be :py:class:˚mne.io.BaseRaw˚ in which case 'fs' and 'filter_settings' will be automatically extracted
        or :py:class:˚mne.io.BaseEpochs˚ in which case 'fs', 'filter_settings', 'duration' and 'overlap' will be automatically extracted.
    fs : int
        The sampling frequency of the data. Can be omitted if data is :py:class:˚mne.io.BaseRaw˚ or :py:class:˚mne.io.BaseEpochs˚.
    band : tuple
        A tuple containing the lower and upper band of the frequency range used to extract (a-)periodic spectra.
    filter_settings : tuple
        A tuple containing the cut-off of the High- and Lowpass filter. 
        It is highly advisable to set this correctly in order to avoid filter artifacts in your evaluated frequency range.
        Can be omitted if data is :py:class:˚mne.io.BaseRaw˚ or :py:class:˚mne.io.BaseEpochs˚.
    duration : float
        The time window of each segment in seconds used to calculate the psd.

    Returns
    -------
    if data : :py:class:`numpy.ndarray`
        freqs : :py:class:`numpy.ndarray`
            The Frequencys associated with the (a-)periodic spectra.
        aperiodic : :py:class:`numpy.ndarray`
            The aperiodic component of the data.
        periodic : :py:class:`numpy.ndarray`
            The periodic component of the data.

    elif data : :py:class:˚mne.io.BaseRaw˚ or :py:class:˚mne.io.BaseEpochs˚
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

    '''

    #set parameters
    assert isinstance(hset, tuple), 'hset should be a tuple of (min, max, step)'
    hset = np.round(np.arange(*hset), 4) 
    nyquist = fs / 2
    hmax = np.max(hset)
    band_evaluated = (band[0] / hmax, band[1] * hmax)

    #TODO: Add safety checks
    assert isinstance(x, np.ndarray), 'Data should be a numpy array.'
    assert x.ndim == 2, 'Data shape needs to be of shape (Channels, Samples).'
    assert band[1] < nyquist, 'Upper band limit must be <  Nyquist (fs / 2).'
    assert band_evaluated[0] > 0, 'The evaluated frequency range is 0 or lower this makes no sense'
    assert band_evaluated[1] < nyquist, 'The evaluated frequency range is higher than Nyquist (fs / 2)'


    #TODO: think about whether we need smoothing or not
    # def _moving_average(x, w):
    #     return np.convolve(x, np.ones(w), 'valid') / w
    # n_avgs=3
    # sgramm_smoother = lambda sgramm, n_avgs : np.array([_moving_average(sgramm[freq,:], w=n_avgs) for freq in range(sgramm.shape[0])])

    #TODO: Allow window definition
    win = dsp.windows.tukey(int(np.floor(fs*duration)), 0.25)
    hop = int(win.shape[0])
    mfft = int(2**np.ceil(np.log2(hmax*hop)))


    SFT = ShortTimeFFT(win, hop=hop, mfft=mfft,
                    fs=fs, scale_to='psd')

    #get time and frequency info
    time = _gen_time_from_sft(SFT, x)
    freq = SFT.f

    sgramm = SFT.spectrogram(x, detr='constant')
    #sgramm_smooth = sgramm_smoother(sgramm, n_avgs)

    sgramms = np.zeros((len(hset), *sgramm.shape))
    #TODO: Only up/ downsampling of specific window
    for i, h in enumerate(hset):

        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator

        # Much faster than FFT-based resampling
        data_up = dsp.resample_poly(x, up, down, axis=-1)
        data_down = dsp.resample_poly(x, down, up, axis=-1)

        # Calculate an up/downsampled version of the PSD using same params as original
        win_up = dsp.windows.tukey(int(fs*duration*h), 0.25)
        hop_up = int(win_up.shape[0])
        SFT_up = ShortTimeFFT(win_up, hop=hop_up, mfft=mfft,
                    fs=int(fs * h), scale_to='psd')
        sgramm_up = SFT_up.spectrogram(data_up, detr='constant')
        #smooth_up = sgramm_smoother(sgramm_up, n_avgs)

        win_dw = dsp.windows.tukey(int(fs*duration/h), 0.25)
        hop_dw = int(win_dw.shape[0])
        SFT_dw = ShortTimeFFT(win_dw, hop=hop_dw, mfft=mfft,
                    fs=int(fs / h), scale_to='psd')
        sgramm_dw = SFT_dw.spectrogram(data_down, detr='constant')
        #smooth_dw = sgramm_smoother(sgramm_dw, n_avgs)

        #subsample the upsampled data in the time domain to allow averaging
        #This is necessary as division by h can cause slight rounding differences that
        #result in actual unintended temporal differences in up/dw for very long segments.
        time_dw = _gen_time_from_sft(SFT_dw, data_down)
        time_up = _gen_time_from_sft(SFT_up, data_up)
        sgramm_ss_up = np.array([_find_nearest(sgramm_up, time_up, t) for t in time])
        sgramm_ss_dw = np.array([_find_nearest(sgramm_dw, time_dw, t) for t in time])

        # geometric mean between up and downsampled
        sgramms[i, :, :] = np.swapaxes(np.swapaxes(np.sqrt(sgramm_ss_up * sgramm_ss_dw), 1, 2), 0, 2)

    sgramm_aperiodic = np.median(sgramms, axis=0)
    sgramm_periodic = sgramm - sgramm_aperiodic

    #NOTE: we need to transpose the data as crop_data extracts stuff from the last axis
    freq, sgramm_aperiodic, sgramm_periodic = _crop_data(band, freq, sgramm_aperiodic, sgramm_periodic, axis=1)

    return sgramm_aperiodic, sgramm_periodic, freq, time


#%%cohrasa
def cohrasa(x, y, fs, hset, kwargs_psd):


    '''
    Function to compute irasa for coherence data - Cohrasa
    '''

    #TODO: add safety checks

    # Calculate the original coherence over the whole data
    freq, cxy = dsp.coherence(x, y, fs=fs, **kwargs_psd)

    # Start the IRASA procedure
    cxys = np.zeros((len(hset), *cxy.shape))

    for i, h in enumerate(hset):
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator
        # resample data
        data_up_x = dsp.resample_poly(x, up, down, axis=-1)
        data_down_x = dsp.resample_poly(x, down, up, axis=-1)

        data_up_y = dsp.resample_poly(y, up, down, axis=-1)
        data_down_y = dsp.resample_poly(y, down, up, axis=-1)

        # Calculate the coherence using same params as original
        _, coh_up = dsp.coherence(data_up_x, data_up_y, h * fs, **kwargs_psd)
        _, coh_down = dsp.coherence(data_down_x, data_down_y, fs / h, **kwargs_psd)
        # Geometric mean of h and 1/h
        cxys[i, :] = np.sqrt(coh_up * coh_down)

    #median resampled data
    Cxy_aperiodic = np.median(cxys, axis=0)
    Cxy_periodic = np.abs(cxy - Cxy_aperiodic)

    return Cxy_periodic, Cxy_aperiodic, freq


#%% temporal cohrasa
