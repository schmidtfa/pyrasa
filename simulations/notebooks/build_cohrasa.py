import numpy as np
import scipy.signal as dsp
import fractions


# %%cohrasa
def cohrasa(x, y, fs, hset, kwargs_psd):
    """
    Function to compute irasa for coherence data - Cohrasa
    """

    # TODO: add safety checks

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

    # median resampled data
    Cxy_aperiodic = np.median(cxys, axis=0)
    Cxy_periodic = np.abs(cxy - Cxy_aperiodic)

    return Cxy_periodic, Cxy_aperiodic, freq


# %% temporal cohrasa
