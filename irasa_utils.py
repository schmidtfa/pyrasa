#%% Utility functions for the IRASA procedures 
import numpy as np    



def _crop_data(band, freqs, psd_aperiodic, psd_periodic, axis):

    ''' Utility function to crop spectra to a defined frequency range '''
    
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=axis)
    psd_periodic = np.compress(~mask_freqs, psd_periodic, axis=axis)

    return freqs, psd_aperiodic, psd_periodic


def _gen_time_from_sft(SFT, x):

    '''Generates time from SFT object'''

    tmin, tmax= SFT.extent(x.shape[-1])[:2]
    delta_t = SFT.delta_t

    time = np.arange(tmin, tmax, delta_t)
    return time


def _find_nearest(sgramm_ud, time_array, time_value):
    
    '''Find the nearest time point in an up/downsampled spectrogram'''

    idx = (np.abs(time_array - time_value)).argmin()

    sgramm_sel = sgramm_ud[:, :, idx]

    return sgramm_sel


def _check_input_data(data, hset, fs, band, filter_settings, duration, overlap):

    '''Check if the input parameters for irasa are specified correctly'''

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
    # check parameters for welch
    overlap /= 100
    assert isinstance(duration, (int, float)), 'You need to set the duration of your time window in seconds'
    assert data.shape[1] > int(fs*duration), 'The duration for each segment cant be longer than the actual data'
    assert np.logical_and(overlap < 1,  overlap > 0), 'The overlap between segments cant be larger than 100% or less than 0%'

    return data, overlap, hset


def _check_input_data_epochs(data, hset, band):

    '''Check if the input parameters for irasa epochs are specified correctly'''

    #check if hset is specified correctly
   
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
