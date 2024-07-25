# %%
import sys
from neurodsp.sim import sim_powerlaw, sim_oscillation
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style('ticks')
sns.set_context('poster')
sys.path.append('/home/schmidtfa/git/SPRiNT')
from scipy.signal import ShortTimeFFT

from pyrasa.irasa import irasa_sprint
from pyrasa.utils.irasa_utils import _gen_time_from_sft

# %% now lets do time resolved irasa
fs = 500
n_seconds = 60
duration = 2
overlap = 0.5


sim_components = {'sim_powerlaw': {'exponent': -1}, 'sim_oscillation': {'freq': 10}}

# x1 = sim_combined(n_seconds=10, fs=fs, components=sim_components)
x1 = sim_oscillation(n_seconds=10, fs=fs, freq=10)  # * .5
x2 = sim_powerlaw(n_seconds=10, fs=fs, exponent=-1)
x3 = sim_oscillation(n_seconds=10, fs=fs, freq=20)
x4 = sim_oscillation(n_seconds=20, fs=fs, freq=30)


x = np.concatenate([x1 + x2, np.concatenate([x2, x2 + x1]) + x4, x2 + x3, x2])
x_2 = np.concatenate([np.concatenate([x2, x2 + x1]) + x4, x1 + x2, x2 + x3, x2])

x = np.tile(x, 10)
x_2 = np.tile(x_2, 10)

x_l = np.concatenate([x[np.newaxis, :], x_2[np.newaxis, :]], axis=0)

# %%
plt.plot(x[:10000])


# %%

win = dsp.windows.tukey(int(fs * duration), 0.25)
SFT = ShortTimeFFT(win, hop=int(win.shape[0]), fs=fs, scale_to='psd')

# get time and frequency info
my_time = _gen_time_from_sft(SFT, x_l)
freqs = SFT.f

sgramm = SFT.spectrogram(x_l, detr='constant')
tmin, tmax = SFT.extent(x_l.shape[-1])[:2]

# %%
plt.loglog(freqs, sgramm[0, :, 2])


# %%
freq_mask = freqs < 40
new_freqs = freqs[freq_mask]

import time

start = time.time()
sgramm_aperiodic, sgramm_periodic, freqs_ap, time_ap = irasa_sprint(
    x_l, fs=fs, band=(1, 100), hop=1, freq_res=0.5, win_duration=2, smooth=False, n_avgs=[3, 7, 11]
)
stop = time.time()
print(stop - start)
# %%
freq_mask_ap = freqs_ap < 40
new_freqs_ap = freqs_ap[freq_mask_ap]

# %%
plt.plot(freqs_ap, sgramm_periodic[0, :, 3])


# %%
for ch in [0, 1]:
    f, axes = plt.subplots(ncols=3, figsize=(12, 4))

    for ax in axes:
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

    axes[0].set_title('Combined \n spectrum')
    axes[0].imshow(
        sgramm[ch, freq_mask, :], extent=(tmin, tmax, new_freqs.min(), new_freqs.max()), origin='lower', aspect='auto'
    )

    axes[1].set_title('Periodic \n spectrum')
    axes[1].imshow(
        sgramm_periodic[ch, freq_mask_ap, :],
        extent=(tmin, tmax, new_freqs_ap.min(), new_freqs_ap.max()),
        origin='lower',
        aspect='auto',
    )

    axes[2].set_title('Aperiodic \n spectrum')
    axes[2].imshow(
        sgramm_aperiodic[ch, freq_mask_ap, :],
        extent=(tmin, tmax, new_freqs_ap.min(), new_freqs_ap.max()),
        origin='lower',
        aspect='auto',
    )

    plt.tight_layout()
# %%
from pyrasa.utils.peak_utils import get_peak_params_sprint, get_band_info

df_peaks = get_peak_params_sprint(sgramm_periodic, freqs=freqs_ap, times=time_ap, min_peak_height=0.2)
df_alpha_peaks = get_band_info(df_peaks, freq_range=(8, 12), ch_names=[0, 1]).query('ch_name == 0')
df_alpha_peaks.head(10)

# %%
from SPRiNT_py import SPRiNT_stft_py, SPRiNT_remove_outliers
# %%


opt = {
    'sfreq': fs,  # Input sampling rate
    'WinLength': duration,  # STFT window length
    'WinOverlap': 0,  # Overlap between sliding windows (in %)
    'WinAverage': 5,  # Number of overlapping windows being averaged
    'rmoutliers': 1,  # Apply peak post-processing
    'maxTime': 6,  # Maximum distance of nearby peaks in time (in n windows)
    'maxFreq': 2.5,  # Maximum distance of nearby peaks in frequency (in Hz)
    'minNear': 3,  # Minimum number of similar peaks nearby (using above bounds)
}

output = SPRiNT_stft_py(x_l, opt)

# %%
for ch in [0, 1]:
    f, axes = plt.subplots(ncols=2, figsize=(8, 4))

    for ax in axes:
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

    axes[0].set_title('Combined \n spectrum (SPRINT)')
    axes[0].imshow(
        output['TF'][ch, :, freq_mask],
        # extent=(tmin, tmax, new_freqs.min(), new_freqs.max()),
        origin='lower',
        aspect='auto',
    )

    axes[1].set_title('Combined \n spectrum (IRASA)')
    axes[1].imshow(
        output['TF'][ch, :, freq_mask],
        # extent=(tmin, tmax, new_freqs.min(), new_freqs.max()),
        origin='lower',
        aspect='auto',
    )

# %%
from fooof import FOOOFGroup, fit_fooof_3d
from fooof.objs.utils import combine_fooofs

start = time.time()
fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.2, max_n_peaks=3)
fgs = fit_fooof_3d(fg, output['freqs'], output['TF'])

stop = time.time()

# %%
print(stop - start)

df_fooof = pd.DataFrame(fgs[0].get_params('gaussian_params'), columns=('CF', 'PW', 'BW', 'Time'))
df_alpha_fooof = df_fooof.query('CF < 12').query('CF > 8')

# %%
plt.plot(df_alpha_peaks['time'], df_alpha_peaks['pw'])
# %%
plt.plot(df_alpha_fooof['Time'], df_alpha_fooof['PW'])
# %%
plt.plot(df_alpha_peaks['time'], df_alpha_peaks['cf'])
# %%
plt.plot(df_alpha_fooof['Time'], df_alpha_fooof['CF'])

# %%
plt.hist(df_alpha_fooof['CF'])
plt.xlim(9.5, 10.5)
# %%
plt.hist(df_alpha_peaks['cf'])
plt.xlim(9.5, 10.5)
# %%

# %%
fgs = [SPRiNT_remove_outliers(fgs[i], output['ts'], opt) for i in range(2)]
# %%
df_alpha_foof_cl = (
    pd.DataFrame(fgs[0].get_params('peak_params'), columns=('CF', 'PW', 'BW', 'Time')).query('CF < 12').query('CF > 8')
)

# %%
plt.plot(df_alpha_peaks['time'], df_alpha_peaks['pw'])
# %%
plt.plot(df_alpha_foof_cl['Time'], df_alpha_foof_cl['PW'])
# %%
plt.plot(df_alpha_peaks['time'], df_alpha_peaks['cf'])
# %%
plt.plot(df_alpha_foof_cl['Time'], df_alpha_foof_cl['CF'])

# %%
pd.DataFrame(fgs[0].get_params('peak_params'))

# %%
print(fgs[0].get_params('peak_params', 'CF'))

# %%
all_fg = combine_fooofs(fgs)

# Explore the results from across all model fits
all_fg.print_results()

# %%
all_fg.report()
# %%
