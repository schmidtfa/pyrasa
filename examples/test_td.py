#%%
from neurodsp.sim import sim_combined
from neurodsp.utils import create_times
import numpy as np
import scipy.signal as dsp


fs = 1000
n_seconds = 60
duration=2
overlap=0.5

sim_components = {'sim_powerlaw': {'exponent' : -1}, 
                  'sim_oscillation': {'freq' : 10}}


sig = sim_combined(n_seconds=n_seconds, fs=fs, components=sim_components)
times = create_times(n_seconds=n_seconds, fs=fs)
#%%
import fractions
kwargs_psd = {'nperseg': duration*fs, 
              'noverlap': duration*fs*overlap}

resampling_factor = 1.5

rat = fractions.Fraction(str(resampling_factor))
up, down = rat.numerator, rat.denominator

# Much faster than FFT-based resampling
data_up = dsp.resample_poly(sig, up, down, axis=-1)
data_down = dsp.resample_poly(sig, down, up, axis=-1)

# Calculate an up/downsampled version of the PSD using same params as original
win_duration = 2
hop = 100

nperseg = int(np.floor(fs * win_duration))

win = dsp.windows.hann(nperseg)

SFT = dsp.ShortTimeFFT(win, hop=hop, fs=fs, scale_to='psd')
t_inc = SFT.T
psd = SFT.spectrogram(sig, detr='constant')

hop_up = int(hop * resampling_factor)
SFT_u = dsp.ShortTimeFFT.from_window('hann', 
                                     nperseg=nperseg, 
                                     fs=fs * resampling_factor, 
                                     noverlap=nperseg-hop_up)
psd_up = SFT_u.stft(data_up)


#%%
hop_dw = int(hop / resampling_factor)
noverlap=nperseg-hop_dw
N = len(data_down)
SFT_d = dsp.ShortTimeFFT.from_window('hann', 
                                     nperseg=nperseg, 
                                     fs=fs / resampling_factor, 
                                     noverlap=nperseg-hop_dw,
                                     fft_mode='centered',)
psd_dw = SFT_d.spectrogram(data_down, p0=0, p1=(N-noverlap)//SFT_d.hop, k_offset=N//2)

# %%
psd_dw.shape

# %%
psd.shape
# %%
psd_up.shape


# %%
import numpy as np
import scipy.signal as dsp
import fractions
from neurodsp.sim import sim_oscillation, sim_powerlaw

# Example signal and parameters
# Set some general settings, to be used across all simulations
fs = 500
n_seconds = 15
duration=4
overlap=0.5

# Create a times vector for the simulations
#times = create_times(n_seconds, fs)


alpha = sim_oscillation(n_seconds=.5, fs=fs, freq=10)
no_alpha = np.zeros(len(alpha))
beta = sim_oscillation(n_seconds=.5, fs=fs, freq=25)
no_beta = np.zeros(len(beta))

exp_1 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-1)
exp_2 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-2)


alphas = np.concatenate([no_alpha, alpha, no_alpha, alpha, no_alpha])
betas = np.concatenate([beta, no_beta, beta, no_beta, beta])

sig = np.concatenate([exp_1 + alphas, 
                         exp_1 + alphas + betas, 
                         exp_1 + betas, 
                         exp_2 + alphas, 
                         exp_2 + alphas + betas, 
                         exp_2 + betas, ])

resampling_factor = 1.1  # Resampling factor
win_duration = 1  # Window duration for spectrogram
hop = 10  # Hop size for spectrogram
hset = np.arange(1, 2, 0.05).round(2)
freq_res = 0.5
nfft = int(fs / freq_res)

# Original spectrogram
# Window and hop size for spectrogram
nperseg = int(np.floor(fs * win_duration))
win = dsp.windows.hann(nperseg)
freq, t, psd = dsp.stft(sig, fs=fs, nfft=nfft, window=win, nperseg=nperseg, 
            noverlap=nperseg - hop, scaling='psd')

psd = (np.abs(psd) ** 2)

average_psd = np.zeros([len(hset), *psd.shape])

for i, resampling_factor in enumerate(hset):
    # Calculate resampling factors
    rat = fractions.Fraction(str(resampling_factor))
    up, down = rat.numerator, rat.denominator

    # Resample the signal
    data_up = dsp.resample_poly(sig, up, down, axis=-1)
    data_down = dsp.resample_poly(sig, down, up, axis=-1)

    # Upsampled spectrogram
    hop_up = int(hop * resampling_factor)
    f_up, t_up, psd_up = dsp.stft(data_up, nfft=nfft, fs=fs * resampling_factor, 
                    window=win, nperseg=nperseg, 
                    noverlap=nperseg - hop_up, scaling='psd')

    # Downsampled spectrogram
    hop_down = int(hop / resampling_factor)
    f_dw, t_dw, psd_dw = dsp.stft(data_down, nfft=nfft, fs=fs / resampling_factor, 
                    window=win, nperseg=nperseg, 
                    noverlap=nperseg - hop_down, scaling='psd')

    # Ensure the time axis has the same number of values by adjusting the hop size
    psd_up = psd_up[:, :psd.shape[1]]
    psd_dw = psd_dw[:, :psd.shape[1]]

    # Average the PSDs
    average_psd[i,:,:] = np.sqrt((np.abs(psd_up) ** 2) * (np.abs(psd_dw) ** 2))
    # If needed, the time axis can be derived from the spectrogram output
    #time_axis = np.linspace(0, len(sig) / fs, psd.shape[1])

    print("Average PSD shape:", average_psd.shape)
    #print("Time axis length:", len(time_axis))

aperiodic = np.median(average_psd, axis=0)
periodic = psd - aperiodic

#%%
from neurodsp.plts import plot_timefrequency#
import matplotlib.pyplot as plt
f, axes = plt.subplots(figsize=(14, 4), ncols=3)

fmask = freq < 50

plot_timefrequency(t, freq[fmask], psd[fmask,:], vmin=0, ax=axes[0])
plot_timefrequency(t, freq[fmask], aperiodic[fmask,:], vmin=0, ax=axes[1])
plot_timefrequency(t, freq[fmask], periodic[fmask,:], vmin=0, ax=axes[2])

# %%
from pyrasa.irasa import irasa_sprint

irasa_sprint_spectrum = irasa_sprint(sig[np.newaxis, :], fs=fs,
                                                       band=(1, 100),
                                                       freq_res=.5,
                                                       hop=100,
                                                       win_duration=1.,
                                                       hset_info=(1.05, 2., 0.05))
# %%
