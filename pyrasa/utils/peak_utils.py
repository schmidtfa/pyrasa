"""Utilities for extracting peak parameters."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.signal as dsp


# %% find peaks irasa style
def get_peak_params(
    periodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    ch_names: Iterable | None = None,
    smooth: bool = True,
    smoothing_window: int | float = 1,
    polyorder: int = 1,
    cut_spectrum: tuple[float, float] | None = None,
    peak_threshold: float = 1.0,
    min_peak_height: float = 0.01,
    peak_width_limits: tuple[float, float] = (0.5, 6.0),
) -> pd.DataFrame:
    """This function can be used to extract peak parameters from the periodic spectrum extracted from IRASA.
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

    """
    if np.isnan(periodic_spectrum).sum() > 0:
        raise ValueError('peak width detection does not work properly with nans')

    freq_step = freqs[1] - freqs[0]

    # generate channel names if not given
    if ch_names is None:
        ch_names = np.arange(periodic_spectrum.shape[0])

    # cut data
    if cut_spectrum is not None:
        freq_range = np.logical_and(freqs > cut_spectrum[0], freqs < cut_spectrum[1])
        freqs = freqs[freq_range]
        periodic_spectrum = periodic_spectrum[:, freq_range]

    # filter signal to get a smoother spectrum for peak extraction
    if smooth:
        window_length = int(smoothing_window // freq_step)
        assert window_length > polyorder, (
            'The smoothing window is too small you either need to increase \n'
            '`smoothing_window` or decrease the `polyorder`.'
        )

        filtered_spectrum = dsp.savgol_filter(periodic_spectrum, window_length=window_length, polyorder=polyorder)
    else:
        filtered_spectrum = periodic_spectrum
    # zero out negative values
    # filtered_spectrum[filtered_spectrum < 0] = 0
    # filtered_spectrum = np.log10(np.abs(filtered_spectrum.min()) + filtered_spectrum + 1e-1)
    # filtered_spectrum += np.abs(filtered_spectrum.min())
    # do peak finding on a channel by channel basis
    peak_list = []
    for ix, ch_name in enumerate(ch_names):
        peaks, peak_dict = dsp.find_peaks(
            filtered_spectrum[ix],
            height=[filtered_spectrum[ix].min(), filtered_spectrum[ix].max()],
            width=peak_width_limits / freq_step,  # in frequency in hz
            prominence=peak_threshold * np.std(filtered_spectrum[ix]),  # threshold in sd
            rel_height=0.75,  # relative peak height based on width
        )

        peak_list.append(
            pd.DataFrame(
                {
                    'ch_name': ch_name,
                    'cf': freqs[peaks],
                    'bw': peak_dict['widths'] * freq_step,
                    'pw': peak_dict['peak_heights'],
                }
            )
        )
    # combine & return
    if len(peak_list) >= 1:
        df_peaks = pd.concat(peak_list)
        # filter for peakheight
        df_peaks = df_peaks.query(f'pw > {min_peak_height}')
    else:
        df_peaks = pd.DataFrame({'ch_name': ch_names, 'cf': np.nan, 'bw': 0, 'pw': 0})
        # 0 is reasonable for power and bandwidth as "no" peaks are detected
        # therefore no power and bandwidth

    return df_peaks


# %% find peaks in irasa sprint


def get_peak_params_sprint(
    periodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    ch_names: Iterable | None = None,
    smooth: bool = True,
    smoothing_window: int | float = 2,
    polyorder: int = 1,
    cut_spectrum: tuple[float, float] | None = None,
    peak_threshold: int | float = 1,
    min_peak_height: float = 0.01,
    peak_width_limits: tuple[float, float] = (0.5, 12),
) -> pd.DataFrame:
    """This function can be used to extract peak parameters from the periodic spectrum extracted from IRASA.
    The algorithm works by smoothing the spectrum, zeroing out negative values and
    extracting peaks based on user specified parameters.

    Parameters: periodic_spectrum : 1d or 2d array
                    Power values for the periodic spectrum extracted using IRASA shape(channel x frequency)
                freqs : 1d array
                    Frequency values for the periodic spectrogram
                time : 1d array
                    time points of the periodic spectrogram
                ch_names: list, optional, default: []
                    Channel names ordered according to the periodic spectrum.
                    If empty channel names are given as numbers in ascending order.
                smoothing window : int, optional, default: 2
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
                    DataFrame containing the center frequency,
                    bandwidth and peak height for each channel and time point.

    """
    time_list = []

    for ix, t in enumerate(times):
        cur_df = get_peak_params(
            periodic_spectrum[:, :, ix],
            freqs,
            ch_names=ch_names,
            smooth=smooth,
            smoothing_window=smoothing_window,
            polyorder=polyorder,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            peak_width_limits=peak_width_limits,
        )
        cur_df['time'] = t
        time_list.append(cur_df)

    df_time = pd.concat(time_list)

    return df_time


# %% find peaks irasa style
def get_band_info(df_peaks: pd.DataFrame, freq_range: tuple[int, int], ch_names: list) -> pd.DataFrame:
    """This function can be used to extract peaks in a specified frequency range
    from the Peak DataFrame obtained via "get_peak_params".

    Parameters : df_peaks : DataFrame
                    DataFrame containing peak parameters obtained via "get_peak_params".
                 freq_range : tuple (int, int)
                    Lower and upper limits for the to be extracted frequency range.
                 ch_names: list
                    Channel names used in the computation of the periodic spectrum.
                    This information is needed to fill channels without a peak in the specified range with nans.

    Returns:    df_band_peaks: DataFrame
                    DataFrame containing the center frequency, bandwidth and peak height for each channel
                    in a specified frequency range

    """
    df_range = df_peaks.query(f'cf > {freq_range[0]}').query(f'cf < {freq_range[1]}')

    # we dont always get a peak in a queried range lets give those channels a nan
    missing_channels = list(set(ch_names).difference(df_range['ch_name'].unique()))
    missing_df = pd.DataFrame(np.nan, index=np.arange(len(missing_channels)), columns=['ch_name', 'cf', 'bw', 'pw'])
    missing_df['ch_name'] = missing_channels

    df_band_peaks = pd.concat([df_range, missing_df]).reset_index().drop(columns='index')

    return df_band_peaks
