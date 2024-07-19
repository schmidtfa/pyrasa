""" Utilities for signal decompositon using IRASA """
import numpy as np    
from copy import copy
import scipy.signal as dsp
from scipy.signal import ShortTimeFFT
from scipy.stats.mstats import gmean


def _crop_data(band, freqs, psd_aperiodic, psd_periodic, axis):

    ''' Utility function to crop spectra to a defined frequency range '''
    
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=axis)
    psd_periodic = np.compress(~mask_freqs, psd_periodic, axis=axis)

    return freqs, psd_aperiodic, psd_periodic


def _gen_time_from_sft(SFT, sgramm):

    '''Generates time from SFT object'''

    tmin, tmax= SFT.extent(sgramm.shape[-1])[:2]
    delta_t = SFT.delta_t

    time = np.arange(tmin, tmax, delta_t)
    return time


def _find_nearest(sgramm_ud, time_array, time_value):
    
    '''Find the nearest time point in an up/downsampled spectrogram'''

    idx = (np.abs(time_array - time_value)).argmin()

    if sgramm_ud.shape[2] >= idx:
        idx = idx - 1 

    sgramm_sel = sgramm_ud[:, :, idx]

    return sgramm_sel


def _get_windows(nperseg, dpss_settings, win_func, win_func_kwargs):
        
        '''Generate a window function used for tapering'''

        win_func_kwargs = copy(win_func_kwargs)
       
        #special settings in case multitapering is required 
        if win_func == dsp.windows.dpss:
            
            time_bandwidth = dpss_settings['time_bandwidth']
            if time_bandwidth < 2.0:
                raise ValueError("time_bandwidth should be >= 2.0 for good tapers")
            
            n_taps = int(np.floor(time_bandwidth - 1))
            win_func_kwargs.update({'NW': time_bandwidth / 2, #half width
                                    'Kmax': n_taps,
                                    'sym': False,
                                    'return_ratios': True})
            win, ratios = win_func(nperseg, **win_func_kwargs)
            if dpss_settings['low_bias']:
                win = win[ratios > 0.9]
                ratios = ratios[ratios > 0.9]
        else:
             win = [win_func(nperseg, **win_func_kwargs)]
             ratios = None

        
        return win, ratios




def _compute_psd_welch(data, fs, irasa_kwargs, spectrum_only=False):

    '''Function to compute power spectral densities using welchs method'''
     
    kwargs2drop = ['h', 'up_down', 'time_orig'] #drop sprint specific keys
    for k in kwargs2drop:
        irasa_kwargs.pop(k, None)

    freq, psd = dsp.welch(data, fs=fs, **irasa_kwargs)

    if spectrum_only:
        return psd
    else: 
        return freq, psd


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


def _check_input_data_mne(data, hset, band):

    '''Check if the input parameters for irasa are specified correctly'''

    #check if hset is specified correctly
    assert isinstance(hset, tuple), 'hset should be a tuple of (min, max, step)'
    
    fs = data.info["sfreq"] 
    filter_settings = (data.info["highpass"], data.info["lowpass"])

    #check that evaluated range fits with the data settings
    nyquist = fs / 2
    hmax = np.max(hset)
    band_evaluated = (band[0] / hmax, band[1] * hmax)
    assert band_evaluated[0] > 0, 'The evaluated frequency range is 0 or lower this makes no sense'
    assert band_evaluated[1] < nyquist, 'The evaluated frequency range is higher than Nyquist (fs / 2)'
    assert np.logical_and(band_evaluated[0] > filter_settings[0], band_evaluated[1] < filter_settings[1]), (f'You run IRASA in a frequency range from {np.round(band_evaluated[0], 2)} - {np.round(band_evaluated[1], 2)}Hz. \n'
                                                                                      f'Your settings specified in "filter_settings" indicate that you have a bandpass filter from {np.round(filter_settings[0], 2)} - {np.round(filter_settings[1], 2)}Hz. \n'
                                                                                      'This means that your evaluated range likely contains filter artifacts. \n'
                                                                                      'Either change your filter settings, adjust hset or the parameter "band" accordingly. \n'
                                                                                     f'You want to make sure that band[0]/hset.max() > {np.round(filter_settings[0], 2)} and that band[1] * hset.max() < {np.round(filter_settings[1], 2)}')


def _check_psd_settings_raw(data_array, fs, duration, overlap):

    '''LEGACY:  Check if the kwargs for welch are specified correctly'''

    # check parameters for welch
    overlap /= 100
    assert isinstance(duration, (int, float)), 'You need to set the duration of your time window in seconds'
    assert data_array.shape[1] > int(fs*duration), 'The duration for each segment cant be longer than the actual data'
    assert np.logical_and(overlap < 1,  overlap > 0), 'The overlap between segments cant be larger than 100% or less than 0%'
