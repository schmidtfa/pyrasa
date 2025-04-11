"""Output Class of pyrasa.irasa_sprint"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define

from pyrasa.utils.aperiodic_utils import compute_aperiodic_model_sprint
from pyrasa.utils.fit_funcs import AbstractFitFun
from pyrasa.utils.peak_utils import get_peak_params_sprint
from pyrasa.utils.types import AperiodicFit

min_ndim = 2


@define
class IrasaTfSpectrum:
    freqs: np.ndarray
    time: np.ndarray
    raw_spectrum: np.ndarray
    aperiodic: np.ndarray
    periodic: np.ndarray
    ch_names: np.ndarray | None

    """
    IrasaTfSpectrum: Output container for time-resolved IRASA spectral decomposition.

    This class encapsulates the output of the `irasa_sprint` algorithm, which performs IRASA-based spectral
    parameterization on time-frequency data (e.g., spectrograms) instead of static power spectral densities.
    It separates each time slice of the spectrogram into periodic and aperiodic components and provides methods
    to further analyze their properties over time.

    The `IrasaTfSpectrum` object offers functionality to:
    - Parametrize the aperiodic component of the time-varying spectrum.
    - Extract spectral peaks (periodic components) at each time point.
    - Compute frequency-resolved error metrics for the aperiodic signal across time.

    Attributes
    ----------
    freqs : np.ndarray
        1D array of frequency bins corresponding to the spectral decomposition.
    time : np.ndarray
        1D array of time bins (center of windowed segments) for the spectrogram.
    raw_spectrum : np.ndarray
        3D array (channels × frequencies × time) of the original spectrogram.
    aperiodic : np.ndarray
        3D array (channels × frequencies × time) of the estimated aperiodic component.
    periodic : np.ndarray
        3D array (channels × frequencies × time) of the residual periodic (oscillatory) component.
    ch_names : np.ndarray or None
        Array of channel names. If None, channels may be indexed numerically.

    Methods
    -------
    fit_aperiodic_model(fit_func='fixed', scale=False, fit_bounds=None)
        Fit a model to the aperiodic spectrogram at each time point to extract parameter trajectories.
    get_peaks(smooth=True, smoothing_window=1, cut_spectrum=None, peak_threshold=2.5, min_peak_height=0.0,
              polyorder=1, peak_width_limits=(0.5, 12))
        Extract peak features from the time-varying periodic spectrum across all time points.
    get_aperiodic_error(peak_kwargs=None)
        Compute residual error of the aperiodic component after removing detected periodic peaks for each time slice.

    Notes
    -----
    The `IrasaTfSpectrum` class can be used to analyze dynamic changes in (neural) time series data,
    such as EEG or MEG, where spectral content varies over time.

    This class complements `IrasaSpectrum`, by allowing for a time-resolved analysis power spectral features.
    The time dimension in this class adds a layer of complexity and opens the door
    for temporally resolved peak detection and aperiodic model fitting.

    Example
    -------
    >>> from pyrasa import irasa_sprint
    >>> tf_spec = irasa_sprint(spectrogram_data, freqs=freqs, times=times, sfreq=1000)
    >>> tf_spec.fit_aperiodic_model(fit_func='knee')
    >>> peaks_df = tf_spec.get_peaks()
    >>> ape_error = tf_spec.get_aperiodic_error(peak_kwargs={'min_peak_height': 0.05})

    """

    def __str__(self) -> str:
        """
        Summary of the IrasaTfSpectrum.
        """
        spec = self.raw_spectrum

        # Determine if we have 2D (single channel) or 3D (multi-channel) input
        arr_shape2d = 2
        arr_shape3d = 3
        if self.raw_spectrum.ndim == arr_shape2d:
            n_channels = 1
        elif self.raw_spectrum.ndim == arr_shape3d:
            n_channels = spec.shape[0]
        else:
            raise ValueError('Invalid spectrum shape. Expected 2D or 3D array.')

        ch_summary = (
            f'{len(self.ch_names)} named channels'
            if self.ch_names is not None
            else f"{n_channels} unnamed channel{'s' if n_channels > 1 else ''}"
        )

        # Frequency info
        freq_min, freq_max = self.freqs[0], self.freqs[-1]
        freq_res = np.mean(np.diff(self.freqs)) if len(self.freqs) > 1 else float('nan')

        # Time info
        time_min, time_max = self.time[0], self.time[-1]
        time_step = np.mean(np.diff(self.time))

        return (
            f'IrasaTfSpectrum Summary\n'
            f'------------------------\n'
            f'Channels      : {ch_summary}\n'
            f'Frequency (Hz): {freq_min:.2f}–{freq_max:.2f} Hz, Δf ≈ {freq_res:.2f} Hz\n'
            f'Time (s)      : {time_min:.2f}–{time_max:.2f} s, Δt ≈ {time_step:.2f} s\n'
            f'Attributes    : raw_spectrum, aperiodic, periodic, freqs, ch_names, time\n'
            f'Methods       : fit_aperiodic_model(), get_peaks(), get_aperiodic_error()\n'
        )

    def fit_aperiodic_model(
        self,
        fit_func: str | type[AbstractFitFun] = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> AperiodicFit:
        """
        Extracts aperiodic parameters from the aperiodic spectrogram using scipy's curve fitting
        function.

        This function computes aperiodic parameters for each time point in the spectrogram by applying either one of
        two different curve fitting functions (`fixed` or `knee`) or a custom function specified by user to the data.
        See examples custom_fit_functions.ipynb. The parameters, along with the goodness of
        fit for each time point, are returned in a concatenated format.

        Parameters
        ----------
        fit_func : str or type[AbstractFitFun], optional
            The fitting function to use. Can be "fixed" for a linear fit or "knee" for a fit that includes a
            knee (bend) in the spectrum or a class that is inherited from AbstractFitFun. The default is 'fixed'..
        scale : bool, optional
            Whether to scale the data to improve fitting accuracy. This is useful when fitting a knee in cases where
            power values are very small, leading to numerical precision issues. Default is False.
        fit_bounds : tuple[float, float] or None, optional
            Tuple specifying the lower and upper frequency bounds for the fit function. If None, the entire frequency
            range is used. Otherwise, the spectrum is cropped to the specified bounds before fitting. Default is None.

        Returns
        -------
        AperiodicFit
            An object containing two pandas DataFrames:
                - aperiodic_params : pd.DataFrame
                    A DataFrame containing the aperiodic parameters (e.g., center frequency, bandwidth, peak height)
                    for each channel and each time point.
                - gof : pd.DataFrame
                    A DataFrame containing the goodness of fit metrics for each channel and each time point.

        Notes
        -----
        This function iterates over each time point in the provided spectrogram to extract aperiodic parameters
        using the specified fit function. It leverages the `compute_aperiodic_model` function for individual fits
        at each time point, then combines the results across all time points into comprehensive DataFrames.

        The `fit_bounds` parameter allows for frequency range restrictions during fitting, which can help in focusing
        the analysis on a particular frequency band of interest.

        Scaling the data using the `scale` parameter can be particularly important when dealing with very small power
        values that might lead to poor fitting due to numerical precision limitations.

        """

        return compute_aperiodic_model_sprint(
            aperiodic_spectrum=self.aperiodic[np.newaxis, :, :] if self.aperiodic.ndim == min_ndim else self.aperiodic,
            freqs=self.freqs,
            times=self.time,
            ch_names=self.ch_names,
            scale=scale,
            fit_func=fit_func,
            fit_bounds=fit_bounds,
        )

    def get_peaks(
        self,
        smooth: bool = True,
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] | None = None,
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 12),
    ) -> pd.DataFrame:
        """
        Extracts peak parameters from a periodic spectrogram obtained via IRASA.

        This method processes a time-resolved periodic spectrum to identify and extract peak parameters such as
        center frequency (cf), bandwidth (bw), and peak height (pw) for each time point. It applies smoothing,
        peak detection, and thresholding according to user-defined parameters
        (see get_peak_params for additional Information).

        Parameters
        ----------
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

        return get_peak_params_sprint(
            periodic_spectrum=self.periodic[np.newaxis, :, :] if self.periodic.ndim == min_ndim else self.periodic,
            freqs=self.freqs,
            times=self.time,
            ch_names=self.ch_names,
            smooth=smooth,
            smoothing_window=smoothing_window,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            polyorder=polyorder,
            peak_width_limits=peak_width_limits,
        )

    def get_aperiodic_error(self, peak_kwargs: None | dict = None) -> np.ndarray:
        """
        Computes the frequency resolved error of the aperiodic spectrum.

        This method first computes the absolute of the periodic spectrum and subsequently zeroes out
        any peaks in the spectrum that are potentially "oscillations", yielding the residual error of the aperiodic
        spectrum as a function of frequency.
        This can be useful when trying to optimize hyperparameters such as the hset.

        peak_kwargs : dict
            A dictionary containing keyword arguments that are passed on to the peak finding method 'get_peaks'

        Returns
        -------
            np.ndarray
                A numpy array containing the frequency resolved squared error of the aperiodic
                spectrum extracted using irasa


        Notes
        -----
        While not strictly necessary, setting peak_kwargs is highly recommended.
        The reason for this is that through up-/downsampling and averaging "broadband"
        parameters such as spectral knees can bleed in the periodic spectrum and could be wrongfully
        interpreted as oscillations. This can be avoided by e.g. explicitely setting `min_peak_height`.
        A good way of making a decision for the periodic parameters is to base it on the settings
        used in peak detection.

        """

        if peak_kwargs is None:
            peak_kwargs = {}

        # get absolute periodic spectrum & zero-out peaks
        freqs = self.freqs
        peaks = self.get_peaks(**peak_kwargs)
        peak_times = peaks['time'].unique()
        ch_names = peaks['ch_name'].unique()

        valid_peak_times = [cur_t in peak_times for cur_t in self.time]
        aperiodic_error = np.abs(self.periodic)
        aperiodic_error_cut = aperiodic_error[:, :, valid_peak_times]

        aperiodic_errors_ch = []
        for c_ix, ch in enumerate(ch_names):
            cur_ch_ape = aperiodic_error_cut[c_ix, :, :]
            cur_peak_ch = peaks.query(f'ch_name == "{ch}"')

            aperiodic_errors_t = []
            for t_ix, cur_t in enumerate(peak_times):
                cur_t_ape = cur_ch_ape[:, t_ix]
                cur_t_ch = cur_peak_ch.query(f'time == "{cur_t}"')

                for _, peak in cur_t_ch.iterrows():
                    cur_upper = peak['cf'] + peak['bw']
                    cur_lower = peak['cf'] - peak['bw']

                    freq_mask = np.logical_and(freqs < cur_upper, freqs > cur_lower)

                    cur_t_ape[freq_mask] = 0

                aperiodic_errors_t.append(cur_t_ape)

            aperiodic_errors_ch.append(np.array(aperiodic_errors_t).T)

        return np.array(aperiodic_errors_ch)

    def plot(
        self,
        cur_ch: None | str | int = None,
        freq_range: tuple[float, float] | None = None,
        time_range: tuple[float, float] | None = None,
        log_x: bool = True,
        vmin: float = 0,
        vmax: float = 0.1,  # noqa
    ) -> None:
        """
        This function plots a raw power spectrum alongside the seperated periodic and aperiodic spectrum.
        The main use of this function is for quality control of the IRASA settings that were used to
        separate periodic and aperiodic spectra.

        NOTE: The y-axis of the raw and aperiodic spectrum is plotted in log-scale
        and the y-axis of the periodic spectrum is always plotted in linear scale. The reason for this decision is
        that the periodic spectrum is the result from subtracting the aperiodic spectrum from the raw powerspectrum.
        This can induce negligible values below 0, very close to 0 or at 0 which creates a weird looking spectrum
        as logging can result in nans, negative values or infs. The main purpose of the
        plotting functionality is to allow you to visually inspect whether something went wrong during
        the IRASA procedure and we feel this is best accomplished using the herein specified settings.

        Parameters
        ----------
        freq_range: tuple of (float, float) or None, optional
            Tuple specifying the frequency range (lower_bound, upper_bound) to which the spectrum should be plotted.
            If None the full frequency range returned from IRASA is plotted. Default is None.
        time_range: tuple of (float, float) or None, optional
            Tuple specifying the selected time range (lower_bound, upper_bound) for which the spectra should be plotted.
            If None the full time range returned from IRASA is plotted. Default is None.
        log_x: bool, optional
            Whether or not to plot the x-axis (Frequencies) log-scaled
        cur_ch: None or int, optional
            The channel index to plot. Per default this is set to None which produces a s
            spectrogram averaged across channels
        """

        upper_rows = [
            'raw',
            'raw',
            'aperiodic',
            'aperiodic',
            'periodic',
            'periodic',
        ]
        lower_rows = [
            'aperiodic spectrum',
            'aperiodic spectrum',
            'aperiodic spectrum',
            'periodic spectrum',
            'periodic spectrum',
            'periodic spectrum',
        ]

        f, axes = plt.subplot_mosaic(
            [upper_rows, lower_rows],  # type: ignore
            layout='tight',
            width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            height_ratios=[0.5, 1.0],
            figsize=(8, 6),
        )

        times, freqs = self.time, self.freqs

        cur_ch = 0

        if freq_range is not None:
            freq_range_mask = np.logical_and(freqs > freq_range[0], freqs < freq_range[1])
        else:
            freq_range_mask = np.ones_like(freqs) == 1

        if time_range is not None:
            time_range_mask = np.logical_and(times > time_range[0], times < time_range[1])
        else:
            time_range_mask = np.ones_like(times) == 1

        sgramms2plot = [
            np.mean(self.raw_spectrum[:, freq_range_mask, :], axis=0)
            if cur_ch is None
            else self.raw_spectrum[cur_ch, freq_range_mask, :],
            np.mean(self.aperiodic[:, freq_range_mask, :], axis=0)
            if cur_ch is None
            else self.aperiodic[cur_ch, freq_range_mask, :],
            np.mean(self.periodic[:, freq_range_mask, :], axis=0)
            if cur_ch is None
            else self.periodic[cur_ch, freq_range_mask, :],
        ]

        for img, cur_sg in zip(upper_rows[::2], sgramms2plot):
            axes[img].imshow(
                cur_sg,
                extent=(times.min(), times.max(), freqs.min(), freqs.max()),
                aspect='auto',
                origin='lower',
                vmin=vmin,
                vmax=vmax,
            )

            axes[img].set_title(img + ' spectrogram')
            axes[img].set_xlabel('Time (s)')
            axes[img].set_ylabel('Frequency (Hz)')

            if time_range is not None:
                t_ix_start = np.argmax(time_range_mask)
                t_start = times[t_ix_start]
                t_stop = times[t_ix_start + np.argmin(time_range_mask[t_ix_start:])]
                axes[img].axvline(t_start, color='r')
                axes[img].axvline(t_stop, color='r')

        for ix, (img, sgram) in enumerate(zip(lower_rows[::3], sgramms2plot[1:])):
            axes[img].plot(
                freqs[freq_range_mask],
                sgramms2plot[0][:, time_range_mask].mean(axis=1),
                label='original spectrum',
            )
            axes[img].plot(
                freqs[freq_range_mask],
                sgram[:, time_range_mask].mean(axis=1),
                label=img,
                color='r' if ix == 0 else 'g',
                alpha=0.7,
            )
            axes[img].legend()

            axes[img].set_xlabel('Frequency (Hz)')
            axes[img].set_ylabel('Power (a.u.)')
            axes[img].set_title(img if time_range is None else img + f'  \n ({t_start:.2f}–{t_stop:.2f} s)')

            if log_x:
                axes[img].set_xscale('log')

            periodic_plot_ix = 1
            if ix < periodic_plot_ix:  # we dont want to log the yscale of the periodic spectrum
                axes[img].set_yscale('log')
