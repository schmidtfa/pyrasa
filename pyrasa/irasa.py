import scipy.signal as dsp
import fractions
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.stats.mstats import gmean

from pyrasa.utils.irasa_utils import (_crop_data, #_compute_psd_welch,
                                      _get_windows, 
                                      _gen_time_from_sft,
                                      _find_nearest
                                      #_compute_sgramm
                                      )
#TODO: Port to Cython

#%%
def _gen_irasa(data, orig_spectrum, fs, irasa_fun, hset, irasa_kwargs, time=None):

    '''
    This function is implementing the IRASA algorithm using a custom function to 
    compute a power/crossspectral density and returns an "aperiodic spectrum".
    '''

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
        if spectra.ndim == 2:
            spectra[i, :] = np.sqrt(spectrum_up * spectrum_dw)
        if spectra.ndim == 3:
            spectra[i, :, :] = np.sqrt(spectrum_up * spectrum_dw)

    aperiodic_spectrum = np.median(spectra, axis=0)
    periodic_spectrum = orig_spectrum - aperiodic_spectrum
    return orig_spectrum, aperiodic_spectrum, periodic_spectrum

#%% irasa
def irasa(data, fs, band, irasa_kwargs, hset_info=(1., 2., 0.05)):

    '''
    This function can be used to seperate aperiodic from periodic power spectra using the IRASA algorithm (Wen & Liu, 2016).
    This implementation of the IRASA algorithm is based on the yasa.irasa function in (Vallat & Walker, 2021).

    WARNING: This is the raw irasa algorithm that gives you maximal control over all parameters. 
    It will not inform or warn you when your parameter settings are invalid. Use this only if you know what you do!
    Otherwise it is recommend to use the irasa_raw, irasa_epochs or irasa_sprint functions from irasa_mne.

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
    assert isinstance(data, np.ndarray), 'Data should be a numpy array.'
    if data.ndim == 1:
        data = data[np.newaxis, :]
    assert data.ndim == 2, 'Data shape needs to be either of shape (Channels, Samples) or (Samples, ).'
    assert isinstance(hset_info, tuple), 'hset should be a tuple of (min, max, step)'

    hset = np.round(np.arange(*hset_info), 4)
    # Calculate original spectrum
    def _compute_psd_welch(data, fs, 
                       window='hann', nperseg=None, 
                       noverlap=None, nfft=None, 
                       detrend='constant', 
                       return_onesided=True, 
                       scaling='density', 
                       axis=-1, 
                       average='mean',
                       spectrum_only=False,
                       **irasa_kwargs,
                       ):

        '''Function to compute power spectral densities using welchs method'''
            
        # kwargs2drop = ['h', 'up_down', 'time_orig'] #drop sprint specific keys
        # for k in kwargs2drop:
        #     irasa_kwargs.pop(k, None)

        freq, psd = dsp.welch(data, fs=fs,
                            window=window, 
                            nperseg=nperseg, 
                            noverlap=noverlap, 
                            nfft=nfft, 
                            detrend=detrend, 
                            return_onesided=return_onesided, 
                            scaling=scaling, 
                            axis=axis, 
                            average=average)

        if spectrum_only:
            return psd
        else: 
            return freq, psd


    freq, psd = _compute_psd_welch(data, fs=fs, **irasa_kwargs)
    
    psd, psd_aperiodic, psd_periodic = _gen_irasa(data=data,
                                                    orig_spectrum=psd,
                                                    fs=fs,
                                                    irasa_fun=_compute_psd_welch,
                                                    hset=hset,
                                                    irasa_kwargs=irasa_kwargs,
                                                    )

    freq, psd_aperiodic, psd_periodic = _crop_data(band, freq, psd_aperiodic, psd_periodic, axis=-1)

    return freq, psd_aperiodic, psd_periodic
    

#%% irasa sprint
def irasa_sprint(data,
                 fs,
                 band=(1,100),
                 freq_res=.5,
                 win_duration=.4,
                 win_func=dsp.windows.hann,
                 win_func_kwargs = None,
                 dpss_settings_time_bandwidth = 2,
                 dpss_settings_low_bias = True,
                 dpss_eigenvalue_weighting = True,
                 hop=10,
                 smooth=True, 
                 n_avgs=1,
                 hset=(1., 2., 0.05)):

    '''

    This function can be used to seperate aperiodic from periodic power spectra 
    using the IRASA algorithm (Wen & Liu, 2016) in a time resolved manner.
    
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
    win_duration : float
        The time width of window in seconds used to calculate the stffts.
    win_func : :py:class:`scipy.signal.windows`
        The desired window function. Can be any window specified in :py:class:`scipy.signal.windows`. 
        The default is multitapering via dpss with sensible preconfigurations. 
        To change the settings adjust the parameter `win_func_kwargs`.
    win_func_kwargs: dict
        A dictionary containing keyword arguments passed to win_func. 

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

    '''

    #set parameters
    assert isinstance(hset, tuple), 'hset should be a tuple of (min, max, step)'
    hset = np.round(np.arange(*hset), 4) 
    nyquist = fs / 2
    hmax = np.max(hset)
    band_evaluated = (band[0] / hmax, band[1] * hmax)
    if win_func_kwargs is None:
        win_func_kwargs = {}

    #TODO: Add safety checks
    assert isinstance(data, np.ndarray), 'Data should be a numpy array.'
    assert data.ndim == 2, 'Data shape needs to be of shape (Channels, Samples).'
    assert band[1] < nyquist, 'Upper band limit must be <  Nyquist (fs / 2).'
    assert band_evaluated[0] > 0, 'The evaluated frequency range is 0 or lower this makes no sense'
    assert band_evaluated[1] < nyquist, 'The evaluated frequency range is higher than Nyquist (fs / 2)'

    mfft = int(fs / freq_res)
    win_kwargs = {'win_func': win_func, 
                  'win_func_kwargs': win_func_kwargs}
    dpss_settings = {'time_bandwidth': dpss_settings_time_bandwidth,
                     'low_bias' : dpss_settings_low_bias,
                     'eigenvalue_weighting': dpss_eigenvalue_weighting,
                     }

    irasa_kwargs = {'mfft': mfft,
                    'hop': hop,
                    'win_duration': win_duration,
                    'h': None,
                    'up_down': None,
                    'dpss_settings': dpss_settings,
                    'win_kwargs': win_kwargs,
                    'time_orig': None,
                    'smooth': smooth, 
                    'n_avgs': n_avgs}
    
    def _compute_sgramm(x, fs, mfft, hop, win_duration, dpss_settings, win_kwargs, 
               up_down=None, h=None, time_orig=None, smooth=True, n_avgs=3, spectrum_only=False,
               ):
            
            '''Function to compute spectrograms'''

            def _moving_average(x, w):
                 return np.convolve(x, np.ones(w), 'valid') / w
            
            sgramm_smoother = lambda sgramm, n_avgs : np.array([_moving_average(sgramm[freq,:], w=n_avgs) 
                                                                for freq in range(sgramm.shape[0])])
            
            if h is None:
                 nperseg = int(np.floor(fs*win_duration))
            elif np.logical_and(h is not None, up_down == 'up'):
                 nperseg = int(np.floor(fs*win_duration*h))
                 hop = int(hop * h)
            elif np.logical_and(h is not None, up_down == 'down'):
                 nperseg = int(np.floor(fs*win_duration/h))
                 hop = int(hop / h)

            win, ratios = _get_windows(nperseg, dpss_settings, **win_kwargs)

            sgramms = []
            for cur_win in win:
                SFT = ShortTimeFFT(cur_win, hop=hop, mfft=mfft, fs=fs, scale_to='psd')
                cur_sgramm = SFT.spectrogram(x, detr='constant')
                sgramms.append(cur_sgramm)

            if ratios is None:
                sgramm = np.mean(sgramms, axis=0)
            else:
                weighted_sgramms = [ratios[ix] * cur_sgramm for ix, cur_sgramm in enumerate(sgramms)]
                sgramm = np.sum(weighted_sgramms, axis=0) / np.sum(ratios)

            if smooth:
                avgs = []

                n_avgs_r = n_avgs[::-1]
                for avg, avg_r in zip(n_avgs, n_avgs_r):
                 
                    sgramm_fwd = sgramm_smoother(sgramm=np.squeeze(sgramm), n_avgs=avg)[:,avg_r:]
                    sgramm_bwd = sgramm_smoother(sgramm=np.squeeze(sgramm)[:,::-1], n_avgs=avg)[:,::-1][:,avg_r:]
                    sgramm_n = gmean([sgramm_fwd, sgramm_bwd], axis=0)
                    avgs.append(sgramm_n)

                sgramm = np.median(avgs, axis=0)                
                sgramm = sgramm[np.newaxis, :, :]
                
            
            time = _gen_time_from_sft(SFT, x)
            freq = SFT.f[SFT.f > 0]

            #subsample the upsampled data in the time domain to allow averaging
            #This is necessary as division by h can cause slight rounding differences that
            #result in actual unintended temporal differences in up/dw for very long segments.
            if time_orig is not None:
                sgramm = np.array([_find_nearest(sgramm, time, t) for t in time_orig])
                max_t_ix = time_orig.shape[0]
                #swapping axes is necessitated by _find_nearest
                sgramm = np.swapaxes(np.swapaxes(sgramm[:max_t_ix,:,:], 1, 2), 0, 2) #cut time axis for up/downsampled data to allow averaging

            sgramm = np.squeeze(sgramm) #bring in proper format

            if spectrum_only:
                return sgramm
            else: 
                return freq, time, sgramm

    #get time and frequency info
    freq, time, sgramm = _compute_sgramm(data, fs, **irasa_kwargs)

    sgramm, sgramm_aperiodic, sgramm_periodic = _gen_irasa(data=data,
                                                orig_spectrum=sgramm,
                                                fs=fs,
                                                irasa_fun=_compute_sgramm,
                                                hset=hset,
                                                irasa_kwargs=irasa_kwargs,
                                                time=time,
                                                )

    #NOTE: we need to transpose the data as crop_data extracts stuff from the last axis
    freq, sgramm_aperiodic, sgramm_periodic = _crop_data(band, freq, sgramm_aperiodic, sgramm_periodic, axis=0)

    #adjust time info (i.e. cut the padded stuff)
    tmax = data.shape[1] / fs 
    t_mask = np.logical_and(time >= 0, time < tmax)

    return sgramm_aperiodic[:, t_mask], sgramm_periodic[:, t_mask], freq, time[t_mask]


