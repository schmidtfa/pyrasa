"""Output Class of pyrasa.irasa"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define

from pyrasa.utils.aperiodic_utils import compute_aperiodic_model
from pyrasa.utils.fit_funcs import AbstractFitFun
from pyrasa.utils.peak_utils import get_peak_params
from pyrasa.utils.types import AperiodicFit


@define
class IrasaSpectrum:
    freqs: np.ndarray
    raw_spectrum: np.ndarray
    aperiodic: np.ndarray
    periodic: np.ndarray
    ch_names: np.ndarray | None

    """
    IrasaSpectrum: Output container for IRASA-derived spectral data.

    This class encapsulates the output of the IRASA (Irregular Resampling Auto-Spectral Analysis) algorithm,
    providing a structured interface to access and analyze power spectral data. It stores the original (raw)
    power spectrum, as well as the decomposed aperiodic and periodic components for one or more channels.

    The IrasaSpectrum object provides functionality to:
    - Fit a model to the aperiodic spectrum to extract parameters (e.g., slope, offset, knee).
    - Detect spectral peaks in the periodic spectrum.
    - Quantify residual errors in the aperiodic spectrum after peak removal.

    Attributes
    ----------
    freqs : np.ndarray
        1D array of frequency bins corresponding to the power spectra.
    raw_spectrum : np.ndarray
        2D array (channels × frequencies) of the original power spectrum.
    aperiodic : np.ndarray
        2D array (channels × frequencies) of the estimated aperiodic (arrhythmic) component.
    periodic : np.ndarray
        2D array (channels × frequencies) of the residual periodic (oscillatory) component.
    ch_names : np.ndarray or None
        Array of channel names. If None, default channel identifiers may be used.

    Methods
    -------
    fit_aperiodic_model(fit_func='fixed', scale=False, fit_bounds=None)
        Fit a parametric model to the aperiodic component using a specified fitting method.
    get_peaks(smoothing_window=1, cut_spectrum=None, peak_threshold=2.5, min_peak_height=0.0,
              polyorder=1, peak_width_limits=(0.5, 12))
        Extract peak features from the periodic spectrum, such as center frequency and bandwidth.
    get_aperiodic_error(peak_kwargs=None)
        Compute the residual error of the aperiodic component after removing detected periodic peaks.

    Notes
    -----
    The IrasaSpectrum class is designed to provide downstream access and manipulation of the
    results from the IRASA algorithm, enabling users to assess (neural)
    time series by separating rhythmic from arrhythmic features in the frequency domain.

    Example
    -------
    >>> from pyrasa import irasa
    >>> irasa_spec = irasa(time_series_data, sfreq=1000)
    >>> irasa_spec.fit_aperiodic_model(fit_func='knee')
    >>> peaks = irasa_spec.get_peaks()
    >>> error = irasa_spec.get_aperiodic_error(peak_kwargs={'min_peak_height': 0.05})

    """

    def __str__(self) -> str:
        """
        Summary of the IrasaSpectrum.
        """

        n_channels = len(self.ch_names) if self.ch_names is not None else self.raw_spectrum.shape[0]

        freq_min, freq_max = self.freqs[0], self.freqs[-1]
        freq_res = np.mean(np.diff(self.freqs))

        return (
            f'IrasaSpectrum Summary\n'
            f'----------------------\n'
            f'Channels      : {n_channels}\n'
            f'Frequency (Hz): {freq_min:.2f}–{freq_max:.2f} Hz, Δf ≈ {freq_res:.2f} Hz\n'
            f'Attributes    : raw_spectrum, aperiodic, periodic, freqs, ch_names\n'
            f'Methods       : fit_aperiodic_model(), get_peaks(), get_aperiodic_error()\n'
        )

    def fit_aperiodic_model(
        self,
        fit_func: str | type[AbstractFitFun] = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> AperiodicFit:
        """
        Computes aperiodic parameters from the aperiodic spectrum using scipy's curve fitting function.

        This method can be used to model the aperiodic (1/f-like) component of the power spectrum. Per default,
        users can choose between a fixed or knee model fit or specify their own fit method see examples
        custom_fit_functions.ipynb for an example.
        The method returns the fitted parameters for each channel along with some goodness of fit metrics.

        Parameters
        ----------
        fit_func : str or type[AbstractFitFun], optional
            The fitting function to use. Can be "fixed" for a linear fit or "knee" for a fit that includes a
            knee (bend) in the spectrum or a class that is inherited from AbstractFitFun. The default is 'fixed'.
        ch_names : Iterable or None, optional
            Channel names corresponding to the aperiodic spectrum. If None, channels will be named numerically
            in ascending order. Default is None.
        scale : bool, optional
            Whether to scale the data to improve fitting accuracy. This is useful in cases where
            power values are very small (e.g., 1e-28), which may lead to numerical precision issues during fitting.
            After fitting, the parameters are rescaled to match the original data scale. Default is False.
        fit_bounds : tuple[float, float] or None, optional
            Tuple specifying the lower and upper frequency bounds for the fit function. If None, the entire frequency
            range is used. Otherwise, the spectrum is cropped to the specified bounds. Default is None.

        Returns
        -------
        AperiodicFit
            An object containing two pandas DataFrames:
                - aperiodic_params : pd.DataFrame
                    A DataFrame containing the fitted aperiodic parameters for each channel.
                - gof : pd.DataFrame
                    A DataFrame containing the goodness of fit metrics for each channel.

        Notes
        -----
        This function fits the aperiodic component of the power spectrum using scipy's curve fitting function.
        The fitting can be performed using either a simple linear model ('fixed') or a more complex model
        that includes a "knee" point, where the spectrum bends. The resulting parameters can help in
        understanding the underlying characteristics of the aperiodic component in the data.

        If the `fit_bounds` parameter is used, it ensures that only the specified frequency range is considered
        for fitting, which can be important to avoid fitting artifacts outside the region of interest.

        The `scale` parameter can be crucial when dealing with data that have extremely small values,
        as it helps to mitigate issues related to machine precision during the fitting process.

        The function asserts that the input data are of the correct type and shape, and raises warnings
        if the first frequency value is zero, as this can cause issues during model fitting.
        """
        return compute_aperiodic_model(
            aperiodic_spectrum=self.aperiodic,
            freqs=self.freqs,
            ch_names=self.ch_names,
            scale=scale,
            fit_func=fit_func,
            fit_bounds=fit_bounds,
        )

    def get_peaks(
        self,
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] | None = None,
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 12),
    ) -> pd.DataFrame:
        """
        Extracts peak parameters from the periodic spectrum obtained via IRASA.

        This method identifies and extracts peak parameters such as center frequency (cf), bandwidth (bw),
        and peak height (pw) from a periodic spectrum using scipy's find_peaks function.
        The spectrum can be optionally smoothed prior to the peak detection.

        Parameters
        ----------
        smooth : bool, optional
            Whether to smooth the spectrum before peak extraction. Smoothing can help in reducing noise and
            better identifying peaks. Default is True.
        smoothing_window : int or float, optional
            The size of the smoothing window in Hz, passed to the Savitzky-Golay filter. Default is 1 Hz.
        polyorder : int, optional
            The polynomial order for the Savitzky-Golay filter used in smoothing. The polynomial order must be
            less than the window length. Default is 1.
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
            - 'pw': Peak height (power)


        Notes
        -----
        The function works by first optionally smoothing the periodic spectrum using a Savitzky-Golay filter.
        Then, it performs peak detection using the `scipy.signal.find_peaks` function, taking into account the
        specified peak thresholds and width limits. Peaks that do not meet the minimum height requirement are
        filtered out.

        The `cut_spectrum` parameter can be used to focus peak detection on a specific frequency range, which is
        particularly useful when the region of interest is known in advance.

        """

        return get_peak_params(
            self.periodic,
            self.freqs,
            self.ch_names,
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

        aperiodic_errors = []
        # get absolute periodic spectrum
        for ix in range(self.periodic.shape[0]):
            aperiodic_error = np.abs(self.periodic[ix, :])

            # zero-out peaks
            peaks = self.get_peaks(**peak_kwargs)
            freqs = self.freqs

            for _, peak in peaks.iterrows():
                cur_upper = peak['cf'] + peak['bw']
                cur_lower = peak['cf'] - peak['bw']

                freq_mask = np.logical_and(freqs < cur_upper, freqs > cur_lower)

                aperiodic_error[freq_mask] = 0

            aperiodic_errors.append(aperiodic_error)

        return np.array(aperiodic_errors)

    def plot(
        self,
        freq_range: tuple[float, float] | None = None,
        log_x: bool = True,
        average_chs: bool = False,
        units: str = 'a.u.',
    ) -> None:
        """
        This function plots a raw power spectrum alongside the seperated periodic and aperiodic spectrum.
        The main use of this function is for quality control of the IRASA settings that were used to
        separate periodic and aperiodic spectra.

        NOTE: The y-axis of the aperiodic spectrum is plotted in log-scale
        and the y-axis of the periodic spectrum is plotted in linear scale. The reason for this decision is
        that the periodic spectrum is the result from subtracting the aperiodic spectrum from the raw powerspectrum.
        This can induce negligible values below 0, very close to 0 or at 0 which creates a weird looking spectrum
        as logarithms of these values can result in nans, negative values or infs. The main purpose of the
        plotting functionality is to allow you to visually inspect whether something went wrong during
        the IRASA procedure and we feel this is best accomplished using the herein specified settings.

        Parameters
        ----------
        freq_range: tuple of (float, float) or None, optional
            Tuple specifying the frequency range (lower_bound, upper_bound) to which the spectrum should be plotted.
            If None the full frequency range returned from IRASA is plotted. Default is None.
        log_x: bool, optional
            Whether or not to plot the x-axis (Frequencies) log-scaled
        average_chs: bool, optional
            Whether or not the spectra should be averaged across channels
        units: str, optional
            A string that specifies the units in which the data are plotted. Defaults to (a.u.) i.e. arbitrary units.
        """

        if freq_range is not None:
            freq_range_mask = np.logical_and(self.freqs > freq_range[0], self.freqs < freq_range[1])
        else:
            freq_range_mask = np.ones_like(self.freqs) == 1

        f, axes = plt.subplots(ncols=2, figsize=(8, 4))

        axes[0].plot(
            self.freqs[freq_range_mask],
            self.raw_spectrum[:, freq_range_mask].T.mean(axis=1)
            if average_chs
            else self.raw_spectrum[:, freq_range_mask].T,
            label='original spectrum',
        )
        axes[0].plot(
            self.freqs[freq_range_mask],
            self.aperiodic[:, freq_range_mask].T.mean(axis=1) if average_chs else self.aperiodic[:, freq_range_mask].T,
            label='aperiodic spectrum',
            color='r',
            alpha=0.7,
        )
        axes[0].legend()

        axes[1].plot(
            self.freqs[freq_range_mask],
            self.raw_spectrum[:, freq_range_mask].T.mean(axis=1)
            if average_chs
            else self.raw_spectrum[:, freq_range_mask].T,
            label='original spectrum',
        )

        axes[1].plot(
            self.freqs[freq_range_mask],
            self.periodic[:, freq_range_mask].T.mean(axis=1) if average_chs else self.periodic[:, freq_range_mask].T,
            label='periodic spectrum',
            color='g',
            alpha=0.7,
        )

        axes[1].legend()

        titles = ['Raw X Aperiodic \n PowerSpectrum', 'Raw X Periodic \n PowerSpectrum']  #'Raw \n PowerSpectrum',
        for ix, (ax, title) in enumerate(zip(axes, titles)):
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'Power ({units})')
            ax.set_title(title)

            if log_x:
                ax.set_xscale('log')

            periodic_plot_ix = 1
            if ix < periodic_plot_ix:  # we dont want to log the yscale of the periodic spectrum
                ax.set_yscale('log')

        f.tight_layout()
