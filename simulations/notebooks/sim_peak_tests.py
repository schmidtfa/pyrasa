#%%
from neurodsp.sim import sim_oscillation
import matplotlib.pyplot as plt
from pyrasa.utils.peak_utils import get_peak_params
import scipy.signal as dsp
import numpy as np
# %%
f_range = [1, 250]
n_secs = 5*60
fs = 200
osc_freq = 98
f_range4plot = [80, 110]

ts = sim_oscillation(n_seconds=n_secs, fs=fs, freq=osc_freq)
plt.plot(ts[:100])

#%%
freqs, psd = dsp.welch(ts, fs, nperseg=int(4 * fs))
plt.plot(freqs, psd)

freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
freqs, psd = freqs[freq_logical], psd[freq_logical]
# %%
pe_params = get_peak_params(psd[np.newaxis, :], freqs, min_peak_height=0.1)
pe_params
# %%

freq_logical = np.logical_and(freqs >= f_range4plot[0], freqs <= f_range4plot[1])
plt.plot(freqs[freq_logical], psd[freq_logical])
# %%
