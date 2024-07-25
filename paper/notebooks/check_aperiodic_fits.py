#%%
import sys
from neurodsp.sim import sim_knee
from neurodsp.utils import create_times
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('ticks')
sns.set_context('talk')
sys.path.append('../../')

from irasa import irasa


#%%
# Simulation settings
# Set some general settings, to be used across all simulations
fs = 500
n_seconds = 60
duration=4
overlap=0.5

# Create a times vector for the simulations
times = create_times(n_seconds, fs)

#%%
exponent_1 = 0.
exponents = [0.5, 1, 2, 3]
knee_freq = 15


knee_ap = []
knee_osc = []
for exponent_2 in exponents:

    knee_ap.append(sim_knee(n_seconds, fs, 
                        exponent1=-1*exponent_1, 
                        exponent2=-1*exponent_2, 
                        knee=(knee_freq ** (2*exponent_1 + exponent_2))))
    
    # knee_osc.append(sim_combined(n_seconds, 
    #                             fs, 
    #                             components={'sim_oscillation': {'freq' : 10},
    #                                         'sim_knee': {'exponent1': -0., 
    #                                                     'exponent2':-1*exponent, 
    #                                                     'knee': knee_freq ** exponent}}))

    
# %% plot timeseries

tmax = times < 60
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(times[tmax], knee_ap[0][tmax], label=f'slope = -{exponents[0]}')
ax.plot(times[tmax], knee_ap[1][tmax]+3, label=f'slope = -{exponents[1]}')
ax.plot(times[tmax], knee_ap[2][tmax]+6, label=f'slope = -{exponents[2]}')

sns.despine()
plt.legend()

# %% test irasa
kwargs_psd = {'window': 'hann',
              'nperseg': int(fs*duration), 
              'noverlap': int(fs*duration*overlap)}

# Calculate original spectrum
freq_ap, psds_ap = dsp.welch(knee_ap, fs=fs, **kwargs_psd)
fmax= 150
freq_rasa_ap, psd_aperiodics_ap, psd_periodics_ap = irasa(np.array(knee_ap), band=(0.1, fmax), 
                                                fs=fs, kwargs_psd=kwargs_psd,
                                                hset_info=(1.,2.,.01))


freq_mask = freq_ap < fmax

# fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

# for ix, ax in enumerate(axes):
#     ax.loglog(freq_ap[freq_mask], psds_ap[ix, freq_mask])
#     ax.loglog(freq_rasa_ap, psd_aperiodics_ap[ix])


# %% now lets parametrize the fractal part
from aperiodic_utils import compute_slope, knee_model

#%%
cols = ['Knee', 'Offset', 'Exponent_1', 'Exponent_2']

aps_cmb, gof_cmb = compute_slope(psd_aperiodics_ap, freq_rasa_ap, fit_func='knee')

#%%
aps_cmb

#%%
gof_cmb

#%%
plt.loglog(freq_ap, psds_ap[0], 'ko')
plt.loglog(freq_ap, psds_ap[1], 'ko')
plt.loglog(freq_ap, psds_ap[2], 'ko')
plt.loglog(freq_rasa_ap, 10**knee_model(freq_rasa_ap, *aps_cmb.iloc[0].to_numpy()[:-3]))
plt.loglog(freq_rasa_ap, 10**knee_model(freq_rasa_ap, *aps_cmb.iloc[1].to_numpy()[:-3]))
plt.loglog(freq_rasa_ap, 10**knee_model(freq_rasa_ap, *aps_cmb.iloc[2].to_numpy()[:-3]))
plt.axvline(knee_freq)
#%%

aps1, gof1 = compute_slope(psds_ap[0][np.newaxis, 1:], freq_ap[1:], fit_func='knee')
aps2, gof2 = compute_slope(psds_ap[1][np.newaxis, 1:], freq_ap[1:], fit_func='knee')
aps3, gof3 = compute_slope(psds_ap[2][np.newaxis, 1:], freq_ap[1:], fit_func='knee')

aps_cmb = pd.concat([aps1, aps2, aps3])

aps_cmb
#%%
#%%
plt.loglog(freq_ap, psds_ap[0], 'ko')
plt.loglog(freq_ap, psds_ap[1], 'ko')
plt.loglog(freq_ap, psds_ap[2], 'ko')
plt.loglog(freq_ap, 10**knee_model(freq_ap, *aps1.to_numpy()[0][:-3]))
plt.loglog(freq_ap, 10**knee_model(freq_ap, *aps2.to_numpy()[0][:-3]))
plt.loglog(freq_ap, 10**knee_model(freq_ap, *aps3.to_numpy()[0][:-3]))
plt.axvline(knee_freq)

 
#%%
cur_id = 0
cumsum_psd = np.cumsum(psds_ap[cur_id][1:])
half_pw_freq = freq_ap[1:][np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()]
#exp_guess = np.abs(psds_ap[cur_id][1] / psds_ap[cur_id][-1]) / (np.log10(freq_ap[-1] / freq_ap[1]))
exp_guess = np.log10(psds_ap[cur_id][1] / psds_ap[cur_id][-1])# / 2 #/ np.log10(freq_ap[-1] - freq_ap[1]))
exp_guess

#%%
plt.plot(np.log10(freq_ap[1:]), np.log10(psds_ap[cur_id][1:].min()) * -1 + np.log10(psds_ap[cur_id][1:]))
plt.axvline(np.log10(half_pw_freq))
knee_freq ** (exp_guess)

#%%
from scipy.optimize import curve_fit

def expo_reg(x, a, k, b):

    '''Uses specparams fitting function with a knee'''

    y_hat = a - np.log10(k + x**b)

    return y_hat


def expo_reg_nk(x, a, b):

    '''Uses specparams fitting function without a knee'''

    y_hat = a - np.log10(x**b)

    return y_hat


def knee_model(x, a, k, b1, b2):

    y_hat = a - np.log10(x**b1 * (k + x**b2))

    return y_hat


def mixed_model(x, x1, a1, b1, k, a2, b2, b3):

    '''
    Fit the data using a piecewise function. 
    Where Part A is a fixed model fit and Part B is a fit that allows for a knee.
    Use this to model aperiodic activity that contains a spectral plateau.
    NOTE: This might require some additional testing action
    '''

    condlist = [x < x1,  x > x1]
    funclist = [lambda x: a1 - np.log10(x**b1), 
                lambda x: a1 - np.log10((x-x1)**b2 * (k + (x-x1)**b3))]
    return np.piecewise(x, condlist, funclist)


#%%
cur_id = 2

curv_fit_kwargs = {'maxfev': 5000,
                   'ftol': 1e-5, 
                   'xtol': 1e-5, 
                   'gtol': 1e-5,}
off_guess = [psds_ap[cur_id][1]]
exp_guess = [np.log10(psds_ap[cur_id][1] / psds_ap[cur_id][-1]) - np.log10(freq_ap[-1] / freq_ap[1])]
cumsum_psd = np.cumsum(psds_ap[cur_id][1:])
half_pw_ix = np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()
half_pw_freq = freq_ap[half_pw_ix] 
half_pw_off = [psds_ap[cur_id][half_pw_ix]]
#make the knee guess the point where we have half the power in the spectrum seems plausible to me
knee_guess = [half_pw_freq ** exp_guess[0]] #convert knee freq to knee val
curv_fit_kwargs['p0'] = np.array(knee_guess + off_guess + exp_guess + half_pw_off + knee_guess + exp_guess + exp_guess) #TODO: Really pay attention here
curv_fit_kwargs['bounds'] = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), 
                             (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)) 

p, _ = curve_fit(knee_model, freq_ap[1:], np.log10(psds_ap[cur_id][1:]))
p2, _ = curve_fit(mixed_model, freq_ap[1:], np.log10(psds_ap[cur_id][1:]), **curv_fit_kwargs)

plt.loglog(freq_ap[1:], psds_ap[cur_id][1:], 'ko')
plt.loglog(freq_ap[1:], 10**knee_model(freq_ap[1:], *p))
plt.loglog(freq_ap[1:], 10**mixed_model(freq_ap[1:], *p2))


#%% now do the same with oscillations
from peak_utils import get_peak_params

freq, psds = dsp.welch(knee_osc, fs=fs, **kwargs_psd)
freq_rasa, psd_aperiodics, psd_periodics = irasa(np.array(knee_osc), 
                                                 band=(0.1, 100),
                                                 fs=fs,
                                                 kwargs_psd=kwargs_psd,
                                                 hset_info=(1.,2.,.01))

freq_mask = freq < 100

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

for ix, ax in enumerate(axes):
    ax.loglog(freq[freq_mask], psds[ix, freq_mask])
    ax.loglog(freq_rasa, psd_aperiodics[ix])

get_peak_params(psd_periodics, 
                freq_rasa,
                min_peak_height=0.01,
                peak_width_limits=(0.25, 8),
                peak_threshold=1)

# %%
plt.plot(psd_periodics[0,:])
plt.plot(psd_periodics[1,:])
plt.plot(psd_periodics[2,:])


