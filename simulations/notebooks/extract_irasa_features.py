# %%

from os import listdir
from pathlib import Path
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as dsp
import numpy as np


sns.set_style('ticks')
sns.set_context('poster')


# %%
INDIR = '/mnt/obob/staff/fschmidt/resting_tinnitus/data/data_meg/'
subject_ids = listdir(INDIR)


# for subject_id in subject_ids:

subject_id = subject_ids[2]

cur_data = joblib.load(list(Path(INDIR).glob(f'{subject_id}/{subject_id}__fft_method_irasa.dat'))[0])
freq_range_ap = cur_data['freq'] < 100
freq_range_p = np.logical_and(cur_data['freq'] < 40, cur_data['freq'] > 1)
# %%
plt.loglog(cur_data['freq'][freq_range_ap], cur_data['src']['label_tc_ap'].T[freq_range_ap, :])
# %%
plt.plot(cur_data['freq'][freq_range_p], cur_data['src']['label_tc_p'].T[freq_range_p, :])
# %%


# %% find peaks irasa style
def get_peak_params(
    periodic_spectrum,
    freqs,
    ch_names=[],
    smoothing_window=2,
    cut_spectrum=(1, 40),
    peak_threshold=1,
    peak_width_limits=(1, 12),
):
    """
    This function can be used to extract peak parameters from the periodic spectrum extracted from IRASA.
    The algorithm works by smoothing the spectrum, zeroing out negative values and
    extracting peaks based on user specified parameters.

    Parameters: periodic_spectrum : 1d or 2d array
                    Power values for the periodic spectrum extracted using IRASA shape(channel x frequency)
                freqs : 1d array
                    Frequency values for the periodic spectrum
                ch_names: list, optional, default: []
                    Channel names ordered according to the periodic spectrum.
                    If empty channel names are given as numbers in ascending order.
                smoothing window : int, optional, default: 2
                    Smoothing window in Hz handed over to the savitzky-golay filter.
                cut_spectrum : tuple of (float, float), optional, default (1, 40)
                    Cut the periodic spectrum to limit peak finding to a sensible range
                peak_threshold: float, optional, default: 2.0
                    Relative threshold for detecting peaks. This threshold is defined in
                    relative units of the periodic spectrum
                peak_width_limits : tuple of (float, float), optional, default (1, 12)
                    Limits on possible peak width, in Hz, as (lower_bound, upper_bound)

    Returns:    df_peaks: DataFrame
                    DataFrame containing the center frequency, bandwidth and peak height for each channel


    """

    polyorder = 1  # polyorder for smoothing

    if np.isnan(periodic_spectrum).sum() > 0:
        raise ValueError('peak width detection does not work properly with nans')

    freq_step = freqs[1] - freqs[0]
    window_length = int(smoothing_window // freq_step)

    # generate channel names if not given
    if len(ch_names) == 0:
        ch_names = np.arange(periodic_spectrum.shape[0])

    # cut data
    if cut_spectrum != None:
        freq_range = np.logical_and(freqs > cut_spectrum[0], freqs < cut_spectrum[1])
        freqs = freqs[freq_range]
        periodic_spectrum = periodic_spectrum[:, freq_range]

    # filter signal to get a smoother spectrum for peak extraction
    filtered_spectrum = dsp.savgol_filter(periodic_spectrum, window_length=window_length, polyorder=polyorder)
    # zero out negative values
    filtered_spectrum[filtered_spectrum < 0] = 0

    # do peak finding on a channel by channel basis
    peak_list = []
    for ix, ch_name in enumerate(ch_names):
        peaks, peak_dict = dsp.find_peaks(
            filtered_spectrum[ix],
            width=peak_width_limits / freq_step,  # in frequency in hz
            prominence=peak_threshold * np.std(filtered_spectrum[ix]),
            rel_height=0.75,
        )  # relative peak height based on width
        cur_df = pd.DataFrame(
            {
                'ch_name': 'chx',
                'cf': freqs[peaks],
                'bw': peak_dict['widths'] * freq_step,
                'pw': peak_dict['prominences'],
            }
        )

        peak_list.append(
            pd.DataFrame(
                {
                    'ch_name': ch_name,
                    'cf': freqs[peaks],
                    'bw': peak_dict['widths'] * freq_step,
                    'pw': peak_dict['prominences'],
                }
            )
        )
    # combine & return
    df_peaks = pd.concat(peak_list)

    return df_peaks


# %%
periodic_spectrum = cur_data['src']['label_tc_p']
ch_names = cur_data['src']['label_info']['names_order_mne']
freqs = cur_data['freq']
smoothing_window = 1.5
cut_spectrum = (1, 40)
peak_threshold = 1
peak_width_limits = (0.5, 12)

df_peaks = get_peak_params(
    periodic_spectrum=periodic_spectrum,
    ch_names=ch_names,
    freqs=freqs,
    smoothing_window=smoothing_window,
    cut_spectrum=cut_spectrum,
    peak_threshold=peak_threshold,
    peak_width_limits=peak_width_limits,
)
# %%
polyorder = 1  # polyorder for smoothing
freq_step = freqs[1] - freqs[0]
distance = 1 / freq_step
window_length = int(smoothing_window // freq_step)
# cut data

if cut_spectrum != None:
    freq_range = np.logical_and(freqs > cut_spectrum[0], freqs < cut_spectrum[1])
    freqs = freqs[freq_range]
    periodic_spectrum = periodic_spectrum[:, freq_range]


# filter signal to get a smoother spectrum for peak extraction
filtered_spectrum = dsp.savgol_filter(periodic_spectrum, window_length=window_length, polyorder=polyorder)
# zero out negative values
filtered_spectrum[filtered_spectrum < 0] = 0

# NOTE: peak width cant work with nans -> add a check
ix = 1
peaks, peak_dict = dsp.find_peaks(
    filtered_spectrum[ix],
    width=peak_width_limits / freq_step,  # in frequency in hz
    prominence=peak_threshold * np.std(filtered_spectrum[ix]),
    rel_height=0.75,
)  # relative peak height
# %%
cur_df = pd.DataFrame(
    {'ch_name': 'chx', 'cf': freqs[peaks], 'bw': peak_dict['widths'] * freq_step, 'pw': peak_dict['prominences']}
)

# %%
fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

# orignal
axes[0].plot(freqs, periodic_spectrum[ix, :])
axes[0].set_title('Original Spectrum')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Power (a.u.)')
# filtered
axes[1].plot(freqs, filtered_spectrum[ix, :])
axes[1].set_title('Filtered Spectrum')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power (a.u.)')
# %filtered + peaks
axes[2].plot(freqs, filtered_spectrum[ix, :])
axes[2].set_title('Filtered Spectrum \n+ Peaks')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Power (a.u.)')

for row in cur_df.iterrows():
    logical = np.logical_and(freqs > row[1]['cf'] - row[1]['bw'] / 2, freqs < row[1]['cf'] + row[1]['bw'] / 2)
    axes[2].fill_between(freqs, row[1]['pw'], where=logical, facecolor='red', alpha=0.5)


plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(7, 7))
# %filtered + peaks
ax.plot(freqs, filtered_spectrum[ix, :])
ax.set_title('Filtered Spectrum \n+ Peaks')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (a.u.)')

for row in cur_df.iterrows():
    logical = np.logical_and(freqs > row[1]['cf'] - row[1]['bw'] / 2, freqs < row[1]['cf'] + row[1]['bw'] / 2)
    # logical = np.logical_and(freqs > row[1]['cf'] - 1, freqs < row[1]['cf'] + 1)
    ax.fill_between(freqs, row[1]['pw'], where=logical, facecolor='red', alpha=0.5)


plt.tight_layout()
# %%
peak_dict
# %% now lets get band specific information

freq_range = (8, 12)
ch_names = cur_data['src']['label_info']['names_order_mne']


def get_band_info(df_peaks, freq_range, ch_names):
    """ """

    df_range = df_peaks.query(f'cf > {freq_range[0]}').query(f'cf < {freq_range[1]}')

    # we dont always get a peak in a queried range lets give those channels a nan
    missing_channels = list(set(ch_names).difference(df_range['ch_name'].unique()))
    missing_df = pd.DataFrame(np.nan, index=np.arange(len(missing_channels)), columns=['ch_name', 'cf', 'bw', 'pw'])
    missing_df['ch_name'] = missing_channels

    df_new_range = pd.concat([df_range, missing_df]).reset_index().drop(columns='index')

    return df_new_range


# %%
get_band_info(df_peaks, (8, 12), ch_names)
# %%
