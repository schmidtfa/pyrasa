"""Output Class of pyrasa.irasa_sprint"""

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
