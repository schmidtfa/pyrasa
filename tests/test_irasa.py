#%%
import sys
from neurodsp.sim import sim_combined
from neurodsp.sim import sim_knee, sim_powerlaw, sim_oscillation, sim_variable_oscillation, sim_damped_oscillation
from neurodsp.utils import create_times
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')
sys.path.append('../')

from irasa import irasa, irasa_epochs, irasa_raw


#%%
# Simulation settings
# Set some general settings, to be used across all simulations
fs = 500
n_seconds = 60
duration=2
overlap=0.5

# Create a times vector for the simulations
times = create_times(n_seconds, fs)

#%%
exponents = [1, 2, 3]
knee_freq = 15

#%
knee_ap = []
knee_osc = []
for exponent in exponents:

    knee_ap.append(sim_knee(n_seconds, fs, 
                        exponent1=-0.5, 
                        exponent2=-1*exponent, 
                        knee=knee_freq ** exponent))
    
    knee_osc.append(sim_combined(n_seconds, 
                                fs, 
                                components={'sim_oscillation': {'freq' : 10},
                                            'sim_knee': {'exponent1': -0.5, 
                                                        'exponent2':-1*exponent, 
                                                        'knee': knee_freq ** exponent}}))

    
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
freq, psds = dsp.welch(knee_ap, fs=fs, **kwargs_psd)

psd_aperiodics, psd_periodics, freq_rasa = irasa(np.array(knee_ap), band=(1, 100), fs=fs, kwargs_psd=kwargs_psd)


freq_mask = freq < 100

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

for ix, ax in enumerate(axes):
    ax.loglog(freq[freq_mask], psds[ix, freq_mask])
    ax.loglog(freq_rasa, psd_aperiodics[ix])

#%% now do the same with oscillations
from peak_utils import get_peak_params

freq, psds = dsp.welch(knee_osc, fs=fs, **kwargs_psd)
psd_aperiodics, psd_periodics, freq_rasa = irasa(np.array(knee_osc), 
                                                 band=(1, 100),
                                                 #hplp=(0,100),
                                                 duration=duration, fs=fs)

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
picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False, exclude="bads")
raw.pick(picks)
#%%
aperiodic, periodic = irasa(raw, band=(1, 50), duration=2, return_type=)

#%%
aperiodic.plot();

#%%
aperiodic.plot_topomap();

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

aperiodic, periodic = irasa_epochs(epochs, band=(1, 50), return_type=mne.time_frequency.EpochsSpectrum)

#%%
plt.plot(periodic["Visual/Right"].freqs, periodic["Visual/Left"].get_data().mean(axis=0).T)#.mean(axis=0))
#plt.plot(periodic["Visual/Right"].freqs, periodic["Visual/Right"].get_data().mean(axis=0).mean(axis=0))
#plt.plot(periodic["Visual/Right"].freqs, periodic["Auditory/Left"].get_data().mean(axis=0).mean(axis=0))
#plt.plot(periodic["Visual/Right"].freqs, periodic["Auditory/Right"].get_data().mean(axis=0).mean(axis=0))


plt.show()
#%%
periodic["Visual/Right"].average().plot_topomap();

#%%
epochs.get_data().shape

#%%
cur_trl = epochs.get_data()[0]

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

# %% now lets parametrize the fractal part
from aperiodic_utils import compute_slope

#%%

aps, gof = compute_slope(freq_rasa,  psd_aperiodics[2], fit_func='knee')


#%%
aps
# %%
