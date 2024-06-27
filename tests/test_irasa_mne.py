#%%
import sys
from neurodsp.sim import sim_combined
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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


from irasa import irasa_epochs, irasa_raw

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
aperiodic, periodic, freq = irasa_raw(raw, band=(.25, 50), 
                                duration=2, 
                                hset_info=(1.,2.,.05),
                                as_array=True)

#%%
aperiodic_mne, periodic_mne = irasa_raw(raw, band=(.25, 50), 
                                duration=2, 
                                hset_info=(1.,2.,.05),
                                as_array=False)

#%% inherit from spectrum array
from mne.time_frequency import SpectrumArray

class PeriodicSpectrumArray(SpectrumArray):

    ''' Childclass of SpectrumArray '''

    def plot(self, *, picks=None, average=False, dB=False,
        amplitude=False, xscale="linear", ci="sd",
        ci_alpha=0.3, color="black", alpha=None,
        spatial_colors=True, sphere=None, exclude=(),
        axes=None, show=True,):
        
        super().plot(picks=picks, average=average, dB=dB,
                     amplitude=amplitude, xscale=xscale, ci=ci,
                     ci_alpha=ci_alpha, color=color, alpha=alpha,
                     spatial_colors=spatial_colors, sphere=sphere, exclude=exclude,
                     axes=axes, show=show,)
        
    def plot_topo(self, *, dB=False, layout=None,
        color="w", fig_facecolor="k", axis_facecolor="k",
        axes=None, block=False, show=True,):

        super().plot_topo(dB=dB, layout=layout, color=color, 
                          fig_facecolor=fig_facecolor, axis_facecolor=axis_facecolor,
                          axes=axes, block=block, show=show,)
        
    def get_peaks(self,
                  smoothing_window=1,
                  cut_spectrum=(1, 40),
                  peak_threshold=2.5,
                  min_peak_height=0.0,
                  peak_width_limits=(.5, 12)):

        '''
        This method can be used to extract peak parameters from the periodic spectrum extracted from IRASA.
        The algorithm works by smoothing the spectrum, zeroing out negative values and 
        extracting peaks based on user specified parameters.

        Parameters: smoothing window : int, optional, default: 2
                        Smoothing window in Hz handed over to the savitzky-golay filter.
                    cut_spectrum : tuple of (float, float), optional, default (1, 40)
                        Cut the periodic spectrum to limit peak finding to a sensible range
                    peak_threshold : float, optional, default: 1
                        Relative threshold for detecting peaks. This threshold is defined in 
                        relative units of the periodic spectrum
                    min_peak_height : float, optional, default: 0.01
                        Absolute threshold for identifying peaks. The threhsold is defined in relative
                        units of the power spectrum. Setting this is somewhat necessary when a 
                        "knee" is present in the data as it will carry over to the periodic spctrum in irasa.
                    peak_width_limits : tuple of (float, float), optional, default (.5, 12)
                        Limits on possible peak width, in Hz, as (lower_bound, upper_bound)

        Returns:    df_peaks: DataFrame
                        DataFrame containing the center frequency, bandwidth and peak height for each channel

        '''
        
        from peak_utils import get_peak_params
        peak_df = get_peak_params(self.get_data(),
                                  self.freqs,
                                  self.ch_names,
                                  smoothing_window=smoothing_window,
                                  cut_spectrum=cut_spectrum,
                                  peak_threshold=peak_threshold,
                                  min_peak_height=min_peak_height,
                                  peak_width_limits=peak_width_limits)
        
        return peak_df
    
#%%
psd_array = PeriodicSpectrumArray(periodic, info, freqs=freq)
peak_df = psd_array.get_peaks()
peak_df

#%%
psd_array.plot()

#%%
psd_array.plot_topo()

#%%
psd_array.plot_topomap();

#%%
plt.hist(peak_df['cf'])
#%%
class AperiodicSpectrumArray(SpectrumArray):

    def get_slope(self, fit_func='fixed', fit_bounds=None):

        '''
        This method can be used to extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.
        The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
        as well as the respective goodness of fit.

        Parameters: 
                    fit_func : string
                        Can be either "fixed" or "knee".
                    fit_bounds : None, tuple
                        Lower and upper bound for the fit function, should be None if the whole frequency range is desired.
                        Otherwise a tuple of (lower, upper)

        Returns:    df_aps: DataFrame
                        DataFrame containing the center frequency, bandwidth and peak height for each channel
                    df_gof: DataFrame
                        DataFrame containing the goodness of fit of the specific fit function for each channel.

        '''

        from aperiodic_utils import compute_slope
        df_aps, df_gof = compute_slope(self.get_data(),
                                       self.freqs,
                                       ch_names=self.ch_names,
                                       fit_func=fit_func,
                                       fit_bounds=fit_bounds,
                                       )
        
        return df_aps, df_gof

#%%
psd_array = AperiodicSpectrumArray(aperiodic, info, freqs=freq)
ap_df, df_gof = psd_array.get_slope(fit_func='fixed')
ap_df


#%% note for periodic data the normal plotting function fails
sns.set_style('ticks')
sns.set_context('poster')

f, ax = plt.subplots(figsize=(4,4))
ax.plot(periodic.freqs, periodic.get_data().T);
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')

#%%
#fig, ax = plt.subplots(ncols=5, figsize=(10, 4))
periodic.plot_topomap(dB=False, 
                      vlim=(0, 250),
                      #axes=ax
                      );


#%%
class BaseClass:
    def method(self, param1=10, param2=20, param3=30):
        print(f"param1: {param1}, param2: {param2}, param3: {param3}")

class SubClass(BaseClass):
    def method(self, param1=15, param2=20, param3=30):  # Overwriting param1
        super().method(param1=param1, param2=param2, param3=param3)

# Usage
base_instance = BaseClass()
base_instance.method()  # Outputs: param1: 10, param2: 20, param3: 30

sub_instance = SubClass()
sub_instance.method()












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
