
#%%

#%%

#cols2 = ['Offset', 'Exponent'] 
cur_id = 2
expo_slope_ap, gof = compute_slope_expo(freq_rasa_ap[1:], psd_aperiodics_ap[cur_id][1:], fit_func='exp2')
expo_slope, gof = compute_slope_expo(freq_ap[1:], psds_ap[cur_id][1:], fit_func='exp2')

pd.concat([expo_slope_ap, expo_slope])

#%%
gof

#plt.loglog(freq_ap[1:], expo_reg(freq_ap[1:], *expo_slope[cols2].to_numpy()[0]))
#plt.loglog(freq_ap[1:-2], expo_reg_nk(freq_ap[1:-2], *expo_slope[cols2].to_numpy()[0]))
#plt.loglog(freq_ap[1:], psds_ap[2][1:])
#%%
gof_cmb
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
for ix, ax in enumerate(axes):
    ax.loglog(freq_ap[freq_mask], psds_ap[ix, freq_mask])
    ax.loglog(freq_rasa_ap, psd_aperiodics_ap[ix])



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
