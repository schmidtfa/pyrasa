"""Utilities for extracting peak parameters."""

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.signal as dsp


# %% find peaks irasa style
def get_peak_params(  # noqa C901
    periodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    aperiodic_spectrum: np.ndarray,
    ch_names: Iterable | None = None,
    smooth: bool = True,
    smoothing_window: int | float = 2,
    cut_spectrum: tuple[float, float] | None = None,
    polyorder: int | None = None,
    peak_threshold: float = 1.0,
    min_peak_height: float = 0.0,
    min_peak_distance_hz: float | None = None,
    peak_width_limits: tuple[float, float] = (1.0, 6.0),
) -> pd.DataFrame:
    """
    Extracts peak parameters from the periodic spectrum obtained via IRASA.

    This function identifies and extracts peak parameters such as center frequency (cf), bandwidth (bw),
    and peak height (pw) from a periodic spectrum using scipy's find_peaks function.
    The spectrum can be optionally smoothed prior peak detection.

    Parameters
    ----------
    periodic_spectrum : np.ndarray
        1D or 2D array containing power values of the periodic spectrum (shape: [Channels, Frequencies]
        or [Frequencies]).
    freqs : np.ndarray
        1D array containing frequency values corresponding to the periodic spectrum.
    aperiodic_spectrum : np.ndarray
        Aperiodic power spectrum with the same shape as periodic_spectrum.
    ch_names : Iterable or None, optional
        List of channel names corresponding to the periodic spectrum. If None, channels are labeled numerically.
        Default is None.
    smooth : bool, optional
        Whether to smooth the spectrum before peak extraction. Smoothing can help in reducing noise and
        better identifying peaks. Default is True.
    smoothing_window : int or float, optional
        The size of the smoothing window in Hz, passed to the Savitzky-Golay filter. Default is 1 Hz.
    cut_spectrum : tuple of (float, float) or None, optional
        Tuple specifying the frequency range (lower_bound, upper_bound) to which the spectrum should be cut
        before peak extraction. If None, peaks are detected across the full frequency range. Default is None.
    peak_threshold : float, optional
        Relative threshold for detecting peaks, defined as a multiple of the standard deviation of the
        filtered spectrum. Default is 1.0.
    min_peak_height : float, optional
        The minimum peak height (in absolute units of the power spectrum) required for a peak to be recognized.
        This can be useful for filtering out noise or insignificant peaks, especially when a "knee" is present
        in the original data, which may persist in the periodic spectrum. Default is 0.01.
    min_peak_distance_hz : float, optional
        The minimum distance between two peaks in Hz to avoid fitting too many peaks in close proximity.
    peak_width_limits : tuple of (float, float), optional
        The lower and upper bounds for peak widths, in Hz. This helps in constraining the peak detection to
        meaningful features. Default is (0.5, 12.0).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the detected peak parameters for each channel. The DataFrame includes the
        following columns:
        - 'ch_name': Channel name
        - 'cf': Center frequency of the peak
        - 'bw': Bandwidth of the peak
        - 'pw': Peak height (power) in linear power units
        - 'pw_rel': Log10 peak height relative to the aperiodic background.


    Notes
    -----
    The function works by first optionally smoothing the periodic spectrum using a Savitzky-Golay filter.
    Then, it performs peak detection using the `scipy.signal.find_peaks` function, taking into account the
    specified peak thresholds and width limits. Peaks that do not meet the minimum height requirement are
    filtered out.

    The `cut_spectrum` parameter can be used to focus peak detection on a specific frequency range, which is
    particularly useful when the region of interest is known in advance.

    """

    # constants
    min_n_bins = 2
    max_polyorder = 2
    min_winlen = 5  # odd; typical minimum for SG in this use
    poly_margin = 2  # ensure polyorder <= winlen - POLY_MARGIN

    if np.isnan(periodic_spectrum).sum() > 0:
        raise ValueError('peak width detection does not work properly with nans')

    freq_step = freqs[1] - freqs[0]

    # generate channel names if not given
    if ch_names is None:
        ch_names = [str(i) for i in np.arange(periodic_spectrum.shape[0])]

    # cut data
    if cut_spectrum is not None:
        freq_range = np.logical_and(freqs > cut_spectrum[0], freqs < cut_spectrum[1])
        freqs = freqs[freq_range]
        periodic_spectrum = periodic_spectrum[:, freq_range]
        aperiodic_spectrum = aperiodic_spectrum[:, freq_range]

    # filter signal to get a smoother spectrum for peak extraction
    if smooth:

        def _choose_savgol(
            freq_step: float, smoothing_window_hz: float, polyorder: int | None = None
        ) -> tuple[int, int]:
            def _pick_polyorder(winlen: int) -> int:
                # largest safe order, capped for stability
                return max(1, min(max_polyorder, winlen - poly_margin))

            winlen = int(np.round(smoothing_window_hz / freq_step))
            winlen = max(winlen, min_winlen)
            if winlen % 2 == 0:
                winlen += 1

            if polyorder is None:
                poly = _pick_polyorder(winlen)
            else:
                poly = int(polyorder)
                if poly >= winlen:
                    poly = winlen - poly_margin
                    poly = max(poly, 1)
                    warnings.warn(f'polyorder adjusted to {poly} for window_length={winlen}', RuntimeWarning)

            return winlen, poly

        win, poly = _choose_savgol(freq_step, smoothing_window, polyorder=polyorder)

        filtered_spectrum = dsp.savgol_filter(periodic_spectrum, window_length=win, polyorder=poly, mode='interp')
    else:
        filtered_spectrum = periodic_spectrum

    def _hz_to_bins(val_hz: float, freq_step: float, *, min_bins: int = 1, name: str = 'param') -> int:
        bins = int(np.round(val_hz / freq_step))
        if val_hz > 0 and bins < min_bins:
            warnings.warn(
                f'{name}={val_hz} Hz with freq_step={freq_step} Hz gives {bins} bin(s); '
                f'using {min_bins} bin(s) instead.',
                RuntimeWarning,
            )
            bins = min_bins
        return bins

    dist_bins = 0
    if min_peak_distance_hz and min_peak_distance_hz > 0:
        dist_bins = _hz_to_bins(min_peak_distance_hz, freq_step, min_bins=3, name='min_peak_distance_hz')

    width_bins = np.array(peak_width_limits, dtype=float) / freq_step  # in frequency in hz
    # require >=2 bins min width
    if width_bins[0] < min_n_bins:
        warnings.warn(
            f'peak_width_limits min ({peak_width_limits[0]} Hz) is < 2 bins at freq_step={freq_step}. '
            'Width estimates may be unstable as you are trying to find peaks with a width of 1 bin.',
            RuntimeWarning,
        )

    peak_list = []
    for ix, ch_name in enumerate(ch_names):
        x_det = np.clip(filtered_spectrum[ix], 0, None)
        peaks, peak_dict = dsp.find_peaks(
            x_det,
            distance=dist_bins if dist_bins > 0 else None,
            width=width_bins,
            prominence=peak_threshold * np.std(x_det),  # threshold in sd
            rel_height=0.5,  # relative peak height based on full width half prominence
        )

        ap = aperiodic_spectrum[ix, peaks]

        peak_list.append(
            pd.DataFrame(
                {
                    'ch_name': ch_name,
                    'cf': freqs[peaks],
                    'bw': peak_dict['widths'] * freq_step,
                    'pw': peak_dict['prominences'],
                    'pw_rel': np.log10(ap + x_det[peaks]) - np.log10(ap),
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


# % find peaks in irasa sprint


def get_peak_params_sprint(
    periodic_spectrum: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    aperiodic_spectrum: np.ndarray,
    ch_names: Iterable | None = None,
    smooth: bool = True,
    smoothing_window: int | float = 2,
    polyorder: int | None = None,
    cut_spectrum: tuple[float, float] | None = None,
    peak_threshold: int | float = 2,
    min_peak_height: float = 0.01,
    min_peak_distance_hz: float | None = None,
    peak_width_limits: tuple[float, float] = (1, 6.0),
) -> pd.DataFrame:
    """
    Extracts peak parameters from a periodic spectrogram obtained via IRASA.

    This function processes a time-resolved periodic spectrum to identify and extract peak parameters such as
    center frequency (cf), bandwidth (bw), and peak height (pw) for each time point. It applies smoothing,
    peak detection, and thresholding according to user-defined parameters
    (see get_peak_params for additional Information).

    Parameters
    ----------
    periodic_spectrum : np.ndarray
        2 or 3D array containing power values of the periodic spectrogram (shape: [Channels, Frequencies, Time Points]).
    freqs : np.ndarray
        1D array containing frequency values corresponding to the periodic spectrogram.
    times : np.ndarray
        1D array containing time points corresponding to the periodic spectrogram.
    ch_names : Iterable or None, optional
        List of channel names corresponding to the periodic spectrogram. If None, channels are labeled numerically.
        Default is None.
    smooth : bool, optional
        Whether to smooth the spectrum before peak extraction. Smoothing can help in reducing noise and better
        identifying peaks. Default is True.
    smoothing_window : int or float, optional
        The size of the smoothing window in Hz, passed to the Savitzky-Golay filter. Default is 2 Hz.
    polyorder : int, optional
        The polynomial order for the Savitzky-Golay filter used in smoothing. The polynomial order must be less
        than the window length. Default is 1.
    cut_spectrum : tuple of (float, float) or None, optional
        Tuple specifying the frequency range (lower_bound, upper_bound) to which the spectrum should be cut before
        peak extraction. If None, the full frequency range is used. Default is (1, 40).
    peak_threshold : int or float, optional
        Relative threshold for detecting peaks, defined as a multiple of the standard deviation of the filtered
        spectrum. Default is 1.
    min_peak_height : float, optional
        The minimum peak height (in absolute units of the power spectrum) required for a peak to be recognized.
        This can be useful for filtering out noise or insignificant peaks, especially when a "knee" is present
        in the data. Default is 0.01.
    min_peak_distance_hz : float, optional
        The minimum distance between two peaks in Hz to avoid fitting too many peaks in close proximity.
    peak_width_limits : tuple of (float, float), optional
        The lower and upper bounds for peak widths, in Hz. This helps in constraining the peak detection to
        meaningful features. Default is (0.5, 12).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the detected peak parameters for each channel and time point. The DataFrame
        includes the following columns:
        - 'ch_name': Channel name
        - 'cf': Center frequency of the peak
        - 'bw': Bandwidth of the peak
        - 'pw': Peak height (power)
        - 'time': Corresponding time point for the peak

    Notes
    -----
    This function iteratively processes each time point in the spectrogram, applying the `get_peak_params`
    function to extract peak parameters at each time point. The resulting peak parameters are combined into
    a single DataFrame.

    The function is particularly useful for analyzing time-varying spectral features, such as in dynamic or
    non-stationary M/EEG data, where peaks may shift in frequency, bandwidth, or amplitude over time.

    """

    time_list = []

    for ix, t in enumerate(times):
        cur_df = get_peak_params(
            periodic_spectrum[:, :, ix],
            freqs,
            aperiodic_spectrum=aperiodic_spectrum[:, :, ix],
            ch_names=ch_names,
            smooth=smooth,
            smoothing_window=smoothing_window,
            polyorder=polyorder,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            min_peak_distance_hz=min_peak_distance_hz,
            peak_width_limits=peak_width_limits,
        )
        cur_df['time'] = t
        time_list.append(cur_df)

    df_time = pd.concat(time_list)

    return df_time


# %% find peaks irasa style
def get_band_info(df_peaks: pd.DataFrame, freq_range: tuple[int, int], ch_names: list) -> pd.DataFrame:
    """
    Extract peak information within a specified frequency range from a DataFrame of peak parameters.

    This function filters peaks found in the periodic spectrum or spectrogram to those within
    a specified frequency range.It ensures that every channel is represented in the output,
    filling in missing channels (i.e., channels without detected peaks in the specified range) with NaN values.

    Parameters
    ----------
    df_peaks : pd.DataFrame
        DataFrame containing peak parameters obtained from the `get_peak_params` function.
        The DataFrame should include columns for 'ch_name' (channel name), 'cf' (center frequency),
        'bw' (bandwidth), and 'pw' (peak height).
    freq_range : tuple of (int, int)
        Tuple specifying the lower and upper frequency bounds (in Hz) to filter peaks by. Only peaks
        with center frequencies (cf) within this range will be included in the output.
    ch_names : list
        List of channel names used in the computation of the periodic spectrum. This list ensures that
        every channel is accounted for in the output, even if no peaks were found in the specified range
        for certain channels.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the peak parameters ('cf', 'bw', 'pw') for each channel within the specified
        frequency range. Channels without detected peaks in this range will have NaN values for these parameters.
        The DataFrame includes:
        - 'ch_name': Channel name
        - 'cf': Center frequency of the peak within the specified range
        - 'bw': Bandwidth of the peak within the specified range
        - 'pw': Peak height (power) within the specified range

    Notes
    -----
    This function is useful for isolating and analyzing peaks that occur within specific canonical frequency bands
    (e.g., alpha, beta, gamma) across multiple channels in a periodic spectrum. The inclusion of NaN
    entries for channels without detected peaks ensures that the output DataFrame is complete and aligned
    with the original channel list.

    """

    df_range = df_peaks.query(f'cf > {freq_range[0]}').query(f'cf < {freq_range[1]}')

    # we dont always get a peak in a queried range lets give those channels a nan
    missing_channels = list(set(ch_names).difference(df_range['ch_name'].unique()))
    missing_df = pd.DataFrame(np.nan, index=np.arange(len(missing_channels)), columns=['ch_name', 'cf', 'bw', 'pw'])
    missing_df['ch_name'] = missing_channels

    df_band_peaks = pd.concat([df_range, missing_df]).reset_index().drop(columns='index')

    return df_band_peaks
