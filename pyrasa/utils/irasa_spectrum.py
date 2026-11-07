"""Output Class of pyrasa.irasa"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define

from pyrasa.utils.acf_utils import IrasaACF, psd_to_acf
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
    >>> irasa_spec = irasa(time_series_data, fs=1000, band=(1.0, 100.0))
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
            Object containing three pandas DataFrames:

            - `aperiodic_params`: Fitted model parameters for each channel.
            - `gof`: Goodness-of-fit metrics for each channel.
            - `model`: Predicted aperiodic power values for each channel
            and frequency in the fitted range.

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

    def to_acf(
        self,
        source: str = 'aperiodic',
        max_lag: float = 0.2,
        lag_step: float = 0.001,
    ) -> IrasaACF:
        """Compute a normalized, band-limited ACF from an IRASA spectrum.

        Parameters
        ----------
        source : {'aperiodic', 'raw'}, default 'aperiodic'
            Spectrum to convert. Can either be 'aperiodic' or 'raw' the 'periodic' difference
            spectrum is not a valid target.
        max_lag : float, default 0.2
            Maximum ACF lag, in seconds.
        lag_step : float, default 0.001
            Maximum ACF lag spacing, in seconds.

        Returns
        -------
        IrasaACF
            ACF with channel labels and methods ``half_life`` and ``at_lag``.
        """
        if source not in ('aperiodic', 'raw'):
            raise ValueError("source must be 'aperiodic' or 'raw'.")

        powers = self.aperiodic if source == 'aperiodic' else self.raw_spectrum
        lags, corrs = psd_to_acf(self.freqs, powers, max_lag=max_lag, lag_step=lag_step)
        return IrasaACF(
            lags=lags,
            corrs=np.atleast_2d(corrs),
            ch_names=self.ch_names,
            source=source,
        )

    def get_peaks(
        self,
        smooth: bool = True,
        smoothing_window: int | float = 2,
        polyorder: int | None = None,
        cut_spectrum: tuple[float, float] | None = None,
        peak_threshold: float = 2.0,
        min_peak_height: float = 0.0,
        min_peak_distance_hz: float | None = None,
        peak_width_limits: tuple[float, float] = (1, 6),
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
        smoothing_window : int | float, default 2
            Size of the Savitzky-Golay smoothing window, in Hz.
        polyorder : int | None, optional
            Polynomial order of the Savitzky-Golay filter. If None,
            the order is selected automatically.
        cut_spectrum : tuple of (float, float) or None, optional
            Tuple specifying the frequency range (lower_bound, upper_bound) to which the spectrum should be cut
            before peak extraction. If None, peaks are detected across the full frequency range. Default is None.
        peak_threshold : float, default 2.0
            Minimum peak prominence expressed as a multiple of the
            standard deviation of the processed periodic spectrum.
        min_peak_height : float, default 0.0
            Minimum peak prominence in linear power units. Applied to
            the `pw` values after peak detection.
        min_peak_distance_hz : float or None, optional
            The minimum distance between two peaks in Hz to avoid fitting too many peaks in close proximity.
        peak_width_limits : tuple[float, float], default (1.0, 6.0)
            Minimum and maximum accepted peak widths, in Hz.

        Returns
        -------
        pd.DataFrame
            Detected oscillatory peaks, with one row per peak and the
            following columns:

            - `ch_name`: Name of the channel containing the peak.
            - `cf`: Center frequency of the peak, in Hz.
            - `bw`: Peak bandwidth at half-prominence, in Hz.
            - `pw`: Peak prominence in linear power units.
            - `pw_rel`: Log10 peak height relative to the aperiodic background.

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
            periodic_spectrum=self.periodic,
            freqs=self.freqs,
            aperiodic_spectrum=self.aperiodic,
            ch_names=self.ch_names,
            smooth=smooth,
            smoothing_window=smoothing_window,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            polyorder=polyorder,
            min_peak_distance_hz=min_peak_distance_hz,
            peak_width_limits=peak_width_limits,
        )

    def get_aperiodic_error(self, peak_kwargs: None | dict = None) -> np.ndarray:
        """
        Compute the frequency-resolved residual periodic power after peak removal.

        Calculate the absolute value of the periodic spectrum and set
        frequency bins around detected oscillatory peaks to zero. The
        remaining values provide an estimate of the residual spectral
        power not attributed to detected oscillatory peaks.

        This measure can be used to evaluate IRASA decomposition quality
        and compare different algorithm settings, such as resampling factors.

        Parameters
        ----------
        peak_kwargs : dict | None, optional
            Keyword arguments passed to `get_peaks` for detecting oscillatory
            peaks. If None, the default peak detection settings are used.

        Returns
        -------
        np.ndarray, shape (n_channels, n_freqs)
            Absolute periodic spectral values after masking detected peaks.

        Notes
        -----
        For each detected peak, frequency bins strictly within
        (cf - bw, cf + bw) are set to zero, where cf is the center frequency
        and bw is the estimated peak bandwidth.

        The returned values are absolute residuals, not squared errors.
        """

        error = np.abs(self.periodic)
        peaks = self.get_peaks(**(peak_kwargs or {}))

        ch_names = self.ch_names if self.ch_names is not None else [str(i) for i in range(error.shape[0])]
        ch_index = {ch: ix for ix, ch in enumerate(ch_names)}

        for ch, cf, bw in zip(  # this is done to allow type checking
            peaks['ch_name'],
            peaks['cf'].to_numpy(dtype=float),
            peaks['bw'].to_numpy(dtype=float),
        ):
            freq_mask = (self.freqs > cf - bw) & (self.freqs < cf + bw)
            error[ch_index[str(ch)], freq_mask] = 0

        return error

    def plot(
        self,
        freq_range: tuple[float, float] | None = None,
        log_x: bool = True,
        average_chs: bool = False,
        units: str = 'a.u.',
        legend: bool = True,
    ) -> None:
        """
        This function plots a raw power spectrum alongside the seperated periodic and aperiodic spectrum.
        The main use of this function is for quality control of the IRASA settings that were used to
        separate periodic and aperiodic spectra.

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
        legend: bool, optional
            Whether or not a legend should be plotted

        Notes
        -----
        The left panel displays the original and aperiodic spectra with
        a logarithmic y-axis. The right panel displays the original and
        periodic spectra with a linear y-axis.

        The periodic spectrum can contain negative values because it is
        obtained by subtracting the aperiodic spectrum from the original
        power spectrum. A linear scale is therefore used to preserve
        these values.
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
        if legend:
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
        if legend:
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
