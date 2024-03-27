#%%
import sys
from neurodsp.sim import sim_combined
from neurodsp.sim import sim_knee, sim_powerlaw, sim_oscillation, sim_variable_oscillation, sim_damped_oscillation
from neurodsp.utils import create_times
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('ticks')
sns.set_context('talk')
sys.path.append('../')

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
exponents = [1, 2, 3]
knee_freq = 20

#%
knee_ap = []
knee_osc = []
for exponent in exponents:

    knee_ap.append(sim_knee(n_seconds, fs, 
                        exponent1=-0.0, 
                        exponent2=-1*exponent, 
                        knee=knee_freq ** exponent))
    
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

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

for ix, ax in enumerate(axes):
    ax.loglog(freq_ap[freq_mask], psds_ap[ix, freq_mask])
    ax.loglog(freq_rasa_ap, psd_aperiodics_ap[ix])


# %% now lets parametrize the fractal part
from aperiodic_utils import compute_slope, piecewise_linear, compute_slope_expo

#%%
cols = ['Knee', 'Offset', 'Exponent_1', 'Exponent_2']

aps1, gof1 = compute_slope(freq_rasa_ap,  psd_aperiodics_ap[0], fit_func='knee')
aps2, gof2 = compute_slope(freq_rasa_ap,  psd_aperiodics_ap[1], fit_func='knee')
aps3, gof3 = compute_slope(freq_rasa_ap,  psd_aperiodics_ap[2], fit_func='knee')

aps_cmb = pd.concat([aps1, aps2, aps3])

gof_cmb = pd.concat([gof1, gof2, gof3])

#%%
aps_cmb

#%%

aps1, gof1 = compute_slope(freq_ap[1:],  psds_ap[0][1:], fit_func='knee')
aps2, gof2 = compute_slope(freq_ap[1:],  psds_ap[1][1:], fit_func='knee')
aps3, gof3 = compute_slope(freq_ap[1:],  psds_ap[2][1:], fit_func='knee')

aps_cmb = pd.concat([aps1, aps2, aps3])

aps_cmb
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

def expo_reg2(x, a, b1, k, b2):

    '''Uses specparams fitting function with a knee'''

    y_hat = np.zeros_like(x)

    y_hat = y_hat + a - np.log10((x**b1) * (k + x**b2))

    return y_hat


def compute_slope_expo(freq, psd, fit_func):

    if fit_func == 'knee':
        fit_f = expo_reg
    elif fit_func == 'exp2':
        fit_f = expo_reg2
    elif fit_func == 'fixed':
        fit_f = expo_reg_nk
    
    p, _ = curve_fit(fit_f, freq, psd)

    if fit_func == 'fixed':
        params = pd.DataFrame({'Offset': p[0],
                              'Exponent': p[1]}, 
                               index=[0])

    elif fit_func == 'knee':
        params = pd.DataFrame({'Knee': p[1],
                               'Knee Frequency (Hz)': p[1] ** (1. / p[2]),
                               'Offset': p[0],
                               'Exponent': p[2],
                              }, index=[0])

    elif fit_func == 'exp2':
        params = pd.DataFrame({'Knee': p[2],
                               'Knee Frequency (Hz)': p[2] ** (1. / p[3]),
                               'Offset': p[0],
                               'Exponent_1': p[1],
                               'Exponent_2': p[3],
                              }, index=[0])

    y_log = np.log10(psd)
    residuals = y_log - fit_f(freq, *p)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_log- np.mean(y_log)) ** 2)

    gof = pd.DataFrame({'mse': np.mean(residuals**2),
                        'r_squared': 1 - (ss_res / ss_tot)}, index=[0])

    return params, gof

cols2 = ['Offset', 'Knee', 'Exponent'] 
#cols2 = ['Offset', 'Exponent'] 
cur_id = 2
expo_slope_ap, gof = compute_slope_expo(freq_rasa_ap[1:], np.log10(psd_aperiodics_ap[cur_id][1:]), fit_func='knee')
expo_slope, gof = compute_slope_expo(freq_ap[1:], np.log10(psds_ap[cur_id][1:]), fit_func='knee')

pd.concat([expo_slope_ap, expo_slope])

#%%
import powerlaw
from powerlaw import Fit

fit = Fit( psd_aperiodics_ap[cur_id][1:], xmin=0.25)
fit.power_law.plot_pdf(linestyle=':', color='g')
fit.distribution_compare('power_law', 'exponential')

#%%
psd_aperiodics_ap[cur_id][1:-20] < fit.xmin


#%%
gof

#plt.loglog(freq_ap[1:], expo_reg(freq_ap[1:], *expo_slope[cols2].to_numpy()[0]))
#plt.loglog(freq_ap[1:-2], expo_reg_nk(freq_ap[1:-2], *expo_slope[cols2].to_numpy()[0]))
#plt.loglog(freq_ap[1:], psds_ap[2][1:])
#%%
gof_cmb

#%% plot it
plt.plot(np.log10(freq_ap), np.log10(psds_ap[0]), 'ko')
plt.plot(np.log10(freq_ap), np.log10(psds_ap[1]), 'ko')
plt.plot(np.log10(freq_ap), np.log10(psds_ap[2]), 'ko')
plt.plot(np.log10(freq_rasa_ap), piecewise_linear(np.log10(freq_rasa_ap), *aps1[cols].to_numpy()[0]))
plt.plot(np.log10(freq_rasa_ap), piecewise_linear(np.log10(freq_rasa_ap), *aps2[cols].to_numpy()[0]))
plt.plot(np.log10(freq_rasa_ap), piecewise_linear(np.log10(freq_rasa_ap), *aps3[cols].to_numpy()[0]))
plt.axvline(np.log10(knee_freq))

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





#%%
aps_cmb


















#%% now lets check the mne python implementation
import mne
from mne.datasets import sample

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
event_fname = meg_path / "sample_audvis_filt-0-40_raw-eve.fif"
event_id = {
    "Auditory/Left": 1,
    "Auditory/Right": 2,
    "Visual/Left": 3,
    "Visual/Right": 4,
}
tmin = -0.2
tmax = 0.5

# Load real data as the template
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)
picks = mne.pick_types(raw.info, meg='grad', eeg=False, 
                       stim=False, eog=False, exclude="bads")
raw.pick(picks)
#%%
aperiodic, periodic = irasa_raw(raw, band=(.5, 50), duration=2, hset_info=(1.,2.,.05))

#%%
aperiodic.plot();

aperiodic.plot_topomap();
plt.show()

#%%
aperiodic.plot_topo();

#%% note for periodic data the normal plotting function fails
#%%
periodic.plot(dB=False);

#%%
periodic.plot_topomap(dB=False);

#%% now lets check-out the events
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    #picks=picks,
    baseline=None,
    preload=True,
    verbose=False,
)

#%%

aperiodic, periodic = irasa_epochs(epochs, band=(.5, 50), hset_info=(1.,2.,.05))

#%%
plt.plot(periodic["Visual/Right"].freqs, periodic["Visual/Left"].get_data().mean(axis=0).T)#.mean(axis=0))
#plt.plot(periodic["Visual/Right"].freqs, periodic["Visual/Right"].get_data().mean(axis=0).mean(axis=0))
#plt.plot(periodic["Visual/Right"].freqs, periodic["Auditory/Left"].get_data().mean(axis=0).mean(axis=0))
#plt.plot(periodic["Visual/Right"].freqs, periodic["Auditory/Right"].get_data().mean(axis=0).mean(axis=0))


plt.show()


#%%
periodic["Auditory/Left"].average().plot_topomap(dB=False);
#%%
periodic["Auditory/Right"].average().plot_topomap(dB=False);
#%%
periodic["Visual/Left"].average().plot_topomap(dB=False);
#%%
periodic["Visual/Right"].average().plot_topomap(dB=False);



#%%
aperiodic["Auditory/Left"].average().plot_topomap(dB=False);
#%%
aperiodic["Auditory/Right"].average().plot_topomap(dB=False);
#%%
aperiodic["Visual/Left"].average().plot_topomap(dB=False);
#%%
aperiodic["Visual/Right"].average().plot_topomap(dB=False);

#%%
fs = epochs.info['sfreq']
psd = np.fft.fft(cur_trl) * np.conj(np.fft.fft(cur_trl))
freq_axis = np.fft.fftfreq(len(psd), 1/fs)

#%%

#%%
fft_settings = {'fs': fs,
                'nperseg': cur_trl.shape[1],
                'noverlap': 0,}

f, p_csd2 = dsp.welch(cur_trl, **fft_settings)

#%%
plt.loglog(f, p_w.T)

#%%


#%%
freq_idcs = freq_axis > 0
plt.loglog(freq_axis[freq_idcs], psd[freq_idcs])


#%%

# %%
