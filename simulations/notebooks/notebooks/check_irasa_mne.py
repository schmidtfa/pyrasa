#%%
import sys
from neurodsp.sim import sim_combined
import numpy as np
import seaborn as sns

sns.set_style('ticks')
sns.set_context('paper')
sys.path.append('../')


import matplotlib as mpl
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

import mne
from mne.datasets import sample



from pyrasa.irasa_mne import irasa_raw, irasa_epochs

#%% now lets check the mne python implementation


data_path = sample.data_path()
subjects_dir = data_path / "subjects"
subject = "sample"

meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
fwd = mne.read_forward_solution(fwd_fname)
src = fwd["src"]


raw = mne.io.read_raw_fif(raw_fname)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, 
                       stim=False, eog=False, exclude="bads")
raw.pick(picks)

#%% lets simulate 2 different signals
info = raw.info.copy()
tstep = 1.0 / info["sfreq"]

alpha = sim_combined(60, 
                     info["sfreq"], 
                     components={'sim_oscillation': {'freq' : 10},
                                 'sim_powerlaw': {'exponent': -1.5,}})

beta = sim_combined(60, 
                     info["sfreq"], 
                     components={'sim_oscillation': {'freq' : 20},
                                 'sim_powerlaw': {'exponent': -1.5,}})


#%% now lets simulate some data
#TODO: put in a helper function

def simulate_raw_(signal, scaling_fator, region, subject, info, subjects_dir):
    
    '''Shorthand function to simulate a dipole'''

    selected_label = mne.read_labels_from_annot(
        subject, regexp=region, subjects_dir=subjects_dir,
    )[0]
    location = "center"  # Use the center of the region as a seed.
    extent = 10.0  # Extent in mm of the region.
    label = mne.label.select_sources(
        subject, selected_label, location=location, 
        extent=extent, subjects_dir=subjects_dir
    )

    #this is the source ts
    source_time_series = signal * scaling_fator

    n_events = 1
    events = np.zeros((n_events, 3), int)
    events[:, 0] = np.arange(n_events)  # Events sample.
    events[:, 2] = 1  # All events have the sample id.

    source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
    source_simulator.add_data(label, source_time_series, events)

    raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
    #cov = mne.make_ad_hoc_cov(raw.info)
    #mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])

    return raw

#%% simulate alpha + beta and combine
raw_alpha = simulate_raw_(alpha, 5e-9, "caudalmiddlefrontal-rh", 
                          subject, info, subjects_dir)

raw_beta = simulate_raw_(beta, 10e-10, "caudalmiddlefrontal-lh", 
                          subject, info, subjects_dir)

raw_data = raw_alpha.get_data() + raw_beta.get_data()

raw = mne.io.RawArray(raw_data, info)

picks = mne.pick_types(raw.info, meg='mag', eeg=False, 
                       stim=False, eog=False, exclude="bads")
raw.pick(picks)

#%%
aperiodic_mne, periodic_mne = irasa_raw(raw, band=(.25, 50), 
                                duration=2, 
                                hset_info=(1.,2.,.05),
                                as_array=False)
#%%
df_slopes, df_gof = aperiodic_mne.get_slopes()

#%%
df_slopes.head()
#%%
df_gof.head()

#%%

peak_df = periodic_mne.get_peaks()

peak_df.head()

periodic_mne.plot()

#%% now lets check-out the events
event_id = {
    "Auditory/Left": 1,
    "Auditory/Right": 2,
    "Visual/Left": 3,
    "Visual/Right": 4,
}
tmin = -0.2
tmax = 0.5

# Load real data as the template
event_fname = meg_path / "sample_audvis_filt-0-40_raw-eve.fif"
events = mne.read_events(event_fname)


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

aperiodic, periodic = irasa_epochs(epochs, 
                                   band=(.5, 50), 
                                   hset_info=(1.,2.,.05), 
                                   as_array=False)

#%%
periodic.get_peaks(polyorder=0)

#%%
aperiodic.get_slopes()

#%%
periodic["Auditory/Left"].average().plot_topomap(dB=False)
#%%
periodic["Auditory/Right"].average().plot_topomap(dB=False)
#%%
periodic["Visual/Left"].average().plot_topomap(dB=False)
#%%
periodic["Visual/Right"].average().plot_topomap(dB=False)



#%%
aperiodic["Auditory/Left"].average().plot_topomap(dB=False)
#%%
aperiodic["Auditory/Right"].average().plot_topomap(dB=False)
#%%
aperiodic["Visual/Left"].average().plot_topomap(dB=False)
#%%
aperiodic["Visual/Right"].average().plot_topomap(dB=False)

# %%
