#%%
from pyrasa import irasa
from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as dsp

fs = 1000
n_seconds = 60
duration=4
overlap=0.5
f_range = (1, 100)
sim_components = {'sim_powerlaw': {'exponent' : -1}, 
                  'sim_oscillation': {'freq' : 10}}
sig = sim_combined(n_seconds=n_seconds, fs=fs, components=sim_components)
times = create_times(n_seconds=n_seconds, fs=fs)

#%% psd normal

freq, psd = dsp.welch(sig, fs=fs, nperseg=duration*fs)
freq_cut = np.logical_and(freq >= 1, freq <= 100)

hset = np.arange(1.1, 1.95, 0.05)
hset = np.round(hset, 4)

# The `nperseg` input needs to be set to lock in the size of the FFT's
spectrum_kwargs = {}
spectrum_kwargs['nperseg'] = int(4 * fs)
import fractions

psds = np.zeros((len(hset), *psd.shape))
for ind, h_val in enumerate(hset):

    # Get the up-sampling / down-sampling (h, 1/h) factors as integers
    rat = fractions.Fraction(str(h_val))
    up, dn = rat.numerator, rat.denominator

    # Resample signal
    sig_up = dsp.resample_poly(sig, up, dn, axis=-1)
    sig_dn = dsp.resample_poly(sig, dn, up, axis=-1)

    # Calculate the power spectrum, using the same params as original
    freqs_up, psd_up = dsp.welch(sig_up, h_val * fs, **spectrum_kwargs)
    freqs_dn, psd_dn = dsp.welch(sig_dn, fs / h_val, **spectrum_kwargs)

    # Calculate the geometric mean of h and 1/h
    psds[ind, :] = np.sqrt(psd_up * psd_dn)

# Take the median resampled spectra, as an estimate of the aperiodic component
psd_aperiodic = np.median(psds, axis=0)

# Subtract aperiodic from original, to get the periodic component
psd_periodic = psd - psd_aperiodic

# Apply a relative threshold for tuning which activity is labeled as periodic
thresh = 1
psd_periodic_thresh, psd_aperiodic_thresh = psd_periodic.copy(), psd_aperiodic.copy()
if thresh is not None:
    sub_thresh = np.where(psd_periodic - psd_aperiodic < thresh * np.std(psd))[0]
    psd_periodic_thresh[sub_thresh] = 0
    psd_aperiodic_thresh[sub_thresh] = psd[sub_thresh]

#%%
irasa_out = irasa(sig, 
                    fs=fs, 
                    band=f_range, 
                    psd_kwargs={'nperseg': duration*fs, 
                                'noverlap': duration*fs*overlap
                            },
                    hset_info=(1, 2, 0.05))

#%%
fig, ax = plt.subplots(figsize=(7, 7), ncols=2, nrows=2)


ax[0, 0].loglog(freq[freq_cut], psd_aperiodic[freq_cut], label="NeuroDSP (thresh=None)")
ax[0, 0].loglog(irasa_out.freqs, irasa_out.aperiodic.T, label="PyRASA")

ax[0, 1].plot(freq[freq_cut], psd_periodic[freq_cut], label="NeuroDSP (thresh=None)")
ax[0, 1].plot(irasa_out.freqs, irasa_out.periodic.T, label="PyRASA")
ax[0, 0].set_title("Aperiodic Spectrum")
ax[0, 1].set_title("Periodic Spectrum")
ax[0, 0].legend()
ax[0, 1].legend()


ax[1, 0].loglog(freq[freq_cut], psd_aperiodic_thresh[freq_cut], label="NeuroDSP (thresh=1)")
ax[1, 0].loglog(irasa_out.freqs, irasa_out.aperiodic.T, label="PyRASA")

ax[1, 1].plot(freq[freq_cut], psd_periodic_thresh[freq_cut], label="NeuroDSP (thresh=1)")
ax[1, 1].plot(irasa_out.freqs, irasa_out.periodic.T, label="PyRASA")
ax[1, 0].set_title("Aperiodic Spectrum")
ax[1, 1].set_title("Periodic Spectrum")
ax[1, 0].legend()
ax[1, 1].legend()
plt.tight_layout()



plt.tight_layout()



# %%
