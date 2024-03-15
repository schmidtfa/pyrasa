import scipy.signal as dsp
from scipy.signal import ShortTimeFFT
import fractions
import numpy as np


from irasa_utils import _crop_data, _gen_time_from_sft, _find_nearest
#TODO: Port to Cython

#%% irasa

def irasa(x,                  
          fs,
          band=(1,100),
          duration=4, 
          overlap=0.5, 
          hset=(1.1, 1.95, 0.05)):

    '''
    Function to compute irasa
    '''

    #set parameters
    assert isinstance(hset, tuple), 'hset should be a tuple of (min, max, step)'
    hset = np.round(np.arange(*hset), 4) 
    nyquist = fs / 2
    hmax = np.max(hset)
    band_evaluated = (band[0] / hmax, band[1] * hmax)
    kwargs_psd = {'window': 'hann',
                  'average': 'median',
                  'nperseg': int(fs*duration), 
                  'noverlap': int(fs*duration*overlap)}

    #TODO: Add safety checks
    assert isinstance(x, np.ndarray), 'Data should be a numpy array.'
    assert x.ndim == 2, 'Data shape needs to be of shape (Channels, Samples).'
    assert band[1] < nyquist, 'Upper band limit must be <  Nyquist (fs / 2).'
    assert band_evaluated[0] > 0, 'The evaluated frequency range is 0 or lower this makes no sense'
    assert band_evaluated[1] < nyquist, 'The evaluated frequency range is higher than Nyquist (fs / 2)'


    # Calculate original spectrum
    freq, psd = dsp.welch(x, fs=fs, **kwargs_psd)

    psds = np.zeros((len(hset), *psd.shape))
    for i, h in enumerate(hset):

        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator

        # Much faster than FFT-based resampling
        data_up = dsp.resample_poly(x, up, down, axis=-1)
        data_down = dsp.resample_poly(x, down, up, axis=-1)

        # Calculate an up/downsampled version of the PSD using same params as original
        freqs_up, psd_up = dsp.welch(data_up, fs * h,  **kwargs_psd)
        freqs_dw, psd_dw = dsp.welch(data_down, fs / h, **kwargs_psd)

        # geometric mean between up and downsampled
        psds[i, :] = np.sqrt(psd_up * psd_dw)

    psd_aperiodic = np.median(psds, axis=0)
    psd_periodic = psd - psd_aperiodic

    freq, psd_aperiodic, psd_periodic = _crop_data(band, freq, psd_aperiodic, psd_periodic, axis=-1)

    return psd_aperiodic, psd_periodic, freq



#%% irasa sprint

def irasa_sprint(x,
                 fs,
                 band=(1,100),
                 duration=4,
                 hset=(1.1, 1.95, 0.05)):

    '''
    Function to compute time resolved irasa on a spectrogram.
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
