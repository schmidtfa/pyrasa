# %%
from neurodsp.sim import set_random_seed
from neurodsp.sim import sim_powerlaw, sim_oscillation
from neurodsp.utils import create_times
from neurodsp.plts import plot_timefrequency  #

from neurodsp.timefrequency import compute_wavelet_transform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
freqs = np.arange(1, 50, 0.5)

# %%
mwt = compute_wavelet_transform(
    sim_ts,
    fs=fs,
    freqs=freqs,
    n_cycles=11,
)
# %%
plot_timefrequency(times, freqs, mwt, vmin=0)

# %%
sgramm_ap, sgramm_p, freqs_ir, times_ir = irasa_sprint(
    sim_ts[np.newaxis, :], fs=fs, band=(1, 100), freq_res=0.5, smooth=False, n_avgs=[3, 7, 11]
)


# %% now extract the aperiodic features
from pyrasa.utils.aperiodic_utils import compute_slope_sprint

df_aps, df_gof = compute_slope_sprint(sgramm_ap, freqs=freqs_ir, times=times_ir, fit_func='fixed')


plot_timefrequency(times_ir, freqs_ir, np.squeeze(sgramm_ap), vmin=0)


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
f, ax = plt.subplots(nrows=2, figsize=(8, 4))

ax[0].plot(df_aps['time'], df_aps['Exponent'])
ax[0].set_ylabel('Exponent')
ax[0].set_xlabel('time (s)')
ax[1].plot(df_aps['time'], df_gof['r_squared'])
ax[1].set_ylabel('R2')
ax[1].set_xlabel('time (s)')

f.tight_layout()

# %%
# %%
from pyrasa.utils.peak_utils import get_peak_params_sprint

df_peaks = get_peak_params_sprint(sgramm_p, freqs=freqs_ir, times=times_ir, min_peak_height=0.01)


plot_timefrequency(times_ir, freqs_ir, np.squeeze(sgramm_p), vmin=0)

# %% Plot peak results
f, ax = plt.subplots(nrows=3, figsize=(12, 4))

for ix, cur_key in enumerate(['cf', 'pw', 'bw']):
    ax[ix].plot(df_peaks['time'], df_peaks[cur_key])
    ax[ix].set_ylabel(cur_key)
    ax[ix].set_xlabel('time (s)')

# %%
