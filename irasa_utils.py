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