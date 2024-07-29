#%%
from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import pandas as pd
from pyrasa.irasa import irasa


fs = 500
n_seconds = 60
duration=2
overlap=0.5

sim_components = {'sim_powerlaw': {'exponent' : -1}, 
                  'sim_oscillation': {'freq' : 10}}


sig = sim_combined(n_seconds=n_seconds, fs=fs, components=sim_components)
times = create_times(n_seconds=n_seconds, fs=fs)

max_times = times < 1
f, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].plot(times[max_times], sig[max_times])
axes[0].set_ylabel('Amplitude (a.u.)')
axes[0].set_xlabel('Time (s)')
freq, psd = dsp.welch(sig, fs=fs, nperseg=duration*fs, noverlap=duration*fs*overlap)
axes[1].loglog(freq, psd)
axes[1].set_ylabel('Power (a.u.)')
axes[1].set_xlabel('Frequency (Hz)')

plt.tight_layout()

#from pyrasa.irasa import irasa
#%%
freq_irasa, psd_ap, psd_p = irasa(sig, 
                                  fs=fs, 
                                  band=(1, 100), 
                                  irasa_kwargs={'nperseg': duration*fs, 
                                                'noverlap': duration*fs*overlap
                                            },
                                  hset_info=(1, 2, 0.05))

f, axes = plt.subplots(ncols=2, figsize=(8, 4))
axes[0].set_title('Periodic')
axes[0].plot(freq_irasa, psd_p[0,:])
axes[0].set_ylabel('Power (a.u.)')
axes[0].set_xlabel('Frequency (Hz)')
axes[1].set_title('Aperiodic')
axes[1].loglog(freq_irasa, psd_ap[0,:])
axes[1].set_ylabel('Power (a.u.)')
axes[1].set_xlabel('Frequency (Hz)')

f.tight_layout()

# %% get periodic stuff
from pyrasa.utils.peak_utils import get_peak_params
get_peak_params(psd_p, freqs=freq_irasa)
# %%
from pyrasa.utils.aperiodic_utils import compute_slope

ap_params, gof_params = compute_slope(aperiodic_spectrum=psd_ap,
                                       freqs=freq_irasa-1, 
                                       fit_func='fixed',
                                       #fit_bounds=[0, 40]
                                       )
ap_params
# %%
