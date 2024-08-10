#%%
import numpy as np
import scipy.signal as dsp
from pyrasa.utils.peak_utils import get_peak_params
from neurodsp.sim import sim_powerlaw


fs = 500

sig = sim_powerlaw(n_seconds=60, fs=fs, exponent=-1)
f_range = [0, 100]
#%% test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
freqs, psd = dsp.welch(sig, fs, nperseg=int(4 * fs))
freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
get_peak_params(psd[freq_logical], freqs[freq_logical], min_peak_height=10)
# %%
