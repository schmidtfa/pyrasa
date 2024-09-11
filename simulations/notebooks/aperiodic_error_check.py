#%%
import sys
from neurodsp.sim import set_random_seed
from neurodsp.sim import sim_powerlaw, sim_oscillation
from neurodsp.utils import create_times
from neurodsp.plts import plot_timefrequency#

from neurodsp.timefrequency import compute_wavelet_transform
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

set_random_seed(84)

from pyrasa.irasa import irasa_sprint
# %%
# Set some general settings, to be used across all simulations
fs = 500
n_seconds = 15
duration=4
overlap=0.5

# Create a times vector for the simulations
times = create_times(n_seconds, fs)


alpha = sim_oscillation(n_seconds=.5, fs=fs, freq=10)
no_alpha = np.zeros(len(alpha))
beta = sim_oscillation(n_seconds=.5, fs=fs, freq=25)
no_beta = np.zeros(len(beta))

exp_1 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-1)
exp_2 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-2)


alphas = np.concatenate([no_alpha, alpha, no_alpha, alpha, no_alpha])
betas = np.concatenate([beta, no_beta, beta, no_beta, beta])

sim_ts = np.concatenate([exp_1 + alphas, 
                         exp_1 + alphas + betas, 
                         exp_1 + betas, 
                         exp_2 + alphas, 
                         exp_2 + alphas + betas, 
                         exp_2 + betas, ])

# %%
freqs = np.arange(1, 50, 0.5)
import scipy.signal as dsp

irasa_sprint_spectrum = irasa_sprint(sim_ts,#np.array([sim_ts, sim_ts]), 
                                    fs=fs,
                                    band=(1, 50),
                                    overlap_fraction=.95,
                                    win_duration=.5,
                                    ch_names=['A'],
                                    hset_info=(1.05, 4., 0.05),
                                    win_func=dsp.windows.hann)
# %%
peak_kwargs = { 'smooth': True,
                'smoothing_window':1,
                'peak_threshold':5,
                'min_peak_height':.01,
                'peak_width_limits': (0.5, 12)}
ap_error = irasa_sprint_spectrum.get_aperiodic_error(peak_kwargs)
# %%
plt.plot(ap_error[0,:,:].mean(axis=1))
# %%
