# %%
import sys
from neurodsp.sim import set_random_seed
from neurodsp.sim import sim_powerlaw, sim_oscillation
from neurodsp.utils import create_times
from neurodsp.plts import plot_timefrequency  #

from neurodsp.timefrequency import compute_wavelet_transform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib as mpl

new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none'}
mpl.rcParams.update(new_rc_params)


sns.set_style('ticks')
sns.set_context('talk')

set_random_seed(42)

from pyrasa.irasa import irasa_sprint

# %%
# Set some general settings, to be used across all simulations
fs = 500
n_seconds = 15
duration = 4
overlap = 0.5

# Create a times vector for the simulations
times = create_times(n_seconds, fs)

# %%
alpha = sim_oscillation(n_seconds=0.5, fs=fs, freq=10)
no_alpha = np.zeros(len(alpha))
beta = sim_oscillation(n_seconds=0.5, fs=fs, freq=25)
no_beta = np.zeros(len(beta))

exp_1 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-1)
exp_2 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-2)

# %%
alphas = np.concatenate([no_alpha, alpha, no_alpha, alpha, no_alpha])
betas = np.concatenate([beta, no_beta, beta, no_beta, beta])

sim_ts = np.concatenate(
    [
        exp_1 + alphas,
        exp_1 + alphas + betas,
        exp_1 + betas,
        exp_2 + alphas,
        exp_2 + alphas + betas,
        exp_2 + betas,
    ]
)
# %%
plt.plot(times, sim_ts)
# %%
freqs = np.arange(1, 100, 0.5)

# %%
mwt = compute_wavelet_transform(
    sim_ts,
    fs=fs,
    freqs=freqs,
    n_cycles=11,
)


# %%


# f.savefig('../results/time_freq_sim_mwt.svg')
# %%
sgramm_ap, sgramm_p, freqs_ir, times_ir = irasa_sprint(
    sim_ts[np.newaxis, :], fs=fs, band=(1, 100), freq_res=0.5, smooth=False, n_avgs=[3, 7, 11]
)

# %%
f, axes = plt.subplots(figsize=(14, 4), ncols=3)
mwt = np.abs(mwt)
plot_timefrequency(times, freqs, mwt, vmin=0, ax=axes[0])
plot_timefrequency(times_ir, freqs_ir, np.squeeze(sgramm_ap), vmin=0, ax=axes[1])
plot_timefrequency(times_ir, freqs_ir, np.squeeze(sgramm_p), vmin=0, ax=axes[2])

# f.savefig('../results/time_freq_split.svg')
# %% now extract the aperiodic features
from pyrasa.utils.aperiodic_utils import compute_slope_sprint

df_aps, df_gof = compute_slope_sprint(sgramm_ap[np.newaxis, :, :], freqs=freqs_ir, times=times_ir, fit_func='fixed')

# %%
ave = df_gof['r_squared'].mean()
std = df_gof['r_squared'].std()
print(f'Average goodness of fit {ave} and deviation is {std}')


# %%
ave = df_gof.query('time < 7')['r_squared'].mean()
std = df_gof.query('time < 7')['r_squared'].std()
print(f'Average goodness of fit {ave} and deviation is {std}')


# %%
ave = df_gof.query('time > 7')['r_squared'].mean()
std = df_gof.query('time > 7')['r_squared'].std()
print(f'Average goodness of fit {ave} and deviation is {std}')


# %%
f, ax = plt.subplots(nrows=3, figsize=(8, 7))

ax[0].plot(df_aps['time'], df_aps['Offset'])
ax[0].set_ylabel('Offset')
ax[0].set_xlabel('time (s)')
ax[1].plot(df_aps['time'], df_aps['Exponent'])
ax[1].set_ylabel('Exponent')
ax[1].set_xlabel('time (s)')
ax[2].plot(df_aps['time'], df_gof['r_squared'])
ax[2].set_ylabel('R2')
ax[2].set_xlabel('time (s)')

f.tight_layout()
# f.savefig('../results/time_resolved_exp_r2.svg')
# %%
# %%
from pyrasa.utils.peak_utils import get_peak_params_sprint

df_peaks = get_peak_params_sprint(
    sgramm_p[np.newaxis, :, :], freqs=freqs_ir, times=times_ir, smooth=True, min_peak_height=0.1
)

plot_timefrequency(times_ir, freqs_ir, sgramm_p, vmin=0)

# %% Plot peak results
f, ax = plt.subplots(nrows=3, figsize=(8, 7))

for ix, cur_key in enumerate(['cf', 'pw', 'bw']):
    ax[ix].plot(df_peaks['time'], df_peaks[cur_key])
    ax[ix].set_ylabel(cur_key)
    ax[ix].set_xlabel('time (s)')
    ax[ix].set_xlim(0, 15)

f.tight_layout()
# f.savefig('../results/time_resolved_peak_params.svg')
# %%
sys.path.append('/home/schmidtfa/git/SPRiNT')
from SPRiNT_py import SPRiNT_stft_py
# %%


opt = {
    'sfreq': fs,  # Input sampling rate
    'WinLength': 0.5,  # STFT window length
    'WinOverlap': 50,  # Overlap between sliding windows (in %)
    'WinAverage': 3,  # Number of overlapping windows being averaged
    'rmoutliers': 1,  # Apply peak post-processing
    'maxTime': 6,  # Maximum distance of nearby peaks in time (in n windows)
    'maxFreq': 2.5,  # Maximum distance of nearby peaks in frequency (in Hz)
    'minNear': 3,  # Minimum number of similar peaks nearby (using above bounds)
}

output = SPRiNT_stft_py(sim_ts, opt)

from fooof import FOOOFGroup, fit_fooof_3d


fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.01, max_n_peaks=8)
fgs = fit_fooof_3d(fg, output['freqs'], output['TF'])


df_fooof = pd.DataFrame(fgs[0].get_params('peak_params'), columns=('CF', 'PW', 'BW', 'Time'))
dict_a = dict(zip(df_fooof['Time'].unique().astype(int), output['ts']))
df_fooof['Time'] = df_fooof['Time'].replace(dict_a)
df_fooof['Time'] = df_fooof['Time'] / 2

df_alpha_fooof = df_fooof.query('CF < 12').query('CF > 8')
df_beta_fooof = df_fooof.query('CF < 30').query('CF > 20')
# %%
df_alpha_fooof
# %%
from pyrasa.utils.peak_utils import get_band_info

df_alpha = get_band_info(df_peaks, freq_range=(8, 12), ch_names=[])
df_beta = get_band_info(df_peaks, freq_range=(20, 30), ch_names=[])

# %%
f, ax = plt.subplots(figsize=(12, 4), ncols=2)

ax[0].plot(df_alpha['time'], df_alpha['pw'])
ax[1].plot(df_beta['time'], df_beta['pw'])

yax = ['Alpha Power (8-12Hz)', 'Beta Power (20-30Hz)']
for ix, c_ax in enumerate(ax):
    c_ax.set_xlabel('Time (s)')
    c_ax.set_ylabel(yax[ix])

sns.despine()
f.tight_layout()
# f.savefig('../results/detected_peaks_pyrasa.svg')

# %%
plt.plot(df_beta['time'], df_beta['pw'])


# %%

f, ax = plt.subplots(figsize=(12, 4), ncols=2)

ax[0].plot(df_alpha_fooof['Time'], df_alpha_fooof['PW'])
ax[1].plot(df_beta_fooof['Time'], df_beta_fooof['PW'])

yax = ['Alpha Power (8-12Hz)', 'Beta Power (20-30Hz)']
for ix, c_ax in enumerate(ax):
    c_ax.set_xlabel('Time (s)')
    c_ax.set_ylabel(yax[ix])

sns.despine()
f.tight_layout()
# f.savefig('../results/detected_peaks_sprint.svg')


# %%
df_alpha_fooof
# %%

# %%
