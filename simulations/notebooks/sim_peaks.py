# %%
from neurodsp.sim import sim_powerlaw, sim_peak_oscillation
import numpy as np
import scipy.signal as dsp

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')

from pyrasa.irasa import irasa
from pyrasa.utils.peak_utils import get_peak_params
from pyrasa.utils.aperiodic_utils import compute_slope

# %%
n_secs = 4
fs = 1000

sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=-1)
sig = sim_peak_oscillation(sig_ap=sig_ap, fs=fs, freq=10, bw=3, height=3)

sig_cmb = np.tile(np.concatenate([sig_ap, sig]), 20)

# %%
# osc = sim_oscillation(n_seconds=n_secs, fs=fs, freq=10) + sig_ap

# %%
freq, psd = dsp.welch(x=sig_cmb, fs=fs, nperseg=4 * fs, noverlap=2 * fs)
# freq, psd_osc = dsp.welch(x=osc, fs=fs, nperseg=4*fs, noverlap=2*fs)

f_logical = np.logical_and(freq > 1, freq < 100)

# f_logical = np.logical_and(freq > 8, freq < 12)
f, ax = plt.subplots(figsize=(4, 4))
ax.loglog(freq[f_logical], psd[f_logical])
# ax.loglog(freq[f_logical], psd_osc[f_logical])
# %%
freq_rasa, aperiodic, periodic = irasa(
    sig_cmb,
    fs=fs,
    band=(1, 100),
    kwargs_psd={'nperseg': 4 * fs, 'noverlap': 2 * fs, 'average': 'median'},
    hset_info=(1.0, 4.0, 0.25),
)

# %%
plt.loglog(freq_rasa, aperiodic[0, :])
# %%
df_ap, df_gof = compute_slope(aperiodic, freqs=freq_rasa, fit_func='fixed')

df_ap
# %%
df_gof
# %%

f_logical = np.logical_and(freq_rasa > 1, freq_rasa < 20)
plt.plot(freq_rasa[f_logical], periodic[0, f_logical] * 10)
# f_logical = np.logical_and(freq > 8, freq < 12)
# plt.plot(freq[f_logical], psd[f_logical])

get_peak_params(periodic, freqs=freq_rasa, min_peak_height=0.1)

# %%
