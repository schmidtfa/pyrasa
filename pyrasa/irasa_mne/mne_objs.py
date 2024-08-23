"""Classes for the MNE Python interface."""

# %% inherit from spectrum array

import matplotlib
import mne
import numpy as np
import pandas as pd
from attrs import define
from mne.time_frequency import EpochsSpectrumArray, SpectrumArray

from pyrasa.utils.aperiodic_utils import compute_aperiodic_model
from pyrasa.utils.peak_utils import get_peak_params
from pyrasa.utils.types import AperiodicFit


class PeriodicSpectrumArray(SpectrumArray):
    """Subclass of SpectrumArray"""

    def __init__(
        self: SpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ) -> None:
        # _check_data_shape(data, freqs, info, ndim=2)

        self.__setstate__(
            dict(
                method='IRASA',
                data=data,
                sfreq=info['sfreq'],
                dims=('channel', 'freq'),
                freqs=freqs,
                inst_type_str='Raw',
                data_type='Periodic Power Spectrum',
                info=info,
            )
        )

    def plot(
        self: SpectrumArray,
        *,
        picks: str | np.ndarray | slice | None = None,
        average: bool = False,
        dB: bool = False,  # noqa N803
        amplitude: bool | str = False,
        xscale: str = 'linear',
        ci: float | str | None = 'sd',
        ci_alpha: float = 0.3,
        color: str | tuple = 'black',
        alpha: float | None = None,
        spatial_colors: bool = True,
        sphere: float | np.ndarray | mne.bem.ConductorModel | None | str = None,
        exclude: tuple | list | str = (),
        axes: matplotlib.axes.Axes | list | None = None,
        show: bool = True,
    ) -> None:
        super().plot(
            picks=picks,
            average=average,
            dB=dB,
            amplitude=amplitude,
            xscale=xscale,
            ci=ci,
            ci_alpha=ci_alpha,
            color=color,
            alpha=alpha,
            spatial_colors=spatial_colors,
            sphere=sphere,
            exclude=exclude,
            axes=axes,
            show=show,
        )

    def plot_topo(
        self: SpectrumArray,
        *,
        dB: bool = False,  # noqa N803
        layout: mne.channels.Layout | None = None,
        color: str | tuple = 'w',
        fig_facecolor: str | tuple = 'k',
        axis_facecolor: str | tuple = 'k',
        axes: matplotlib.axes.Axes | list | None = None,
        block: bool = False,
        show: bool = True,
    ) -> None:
        super().plot_topo(
            dB=dB,
            layout=layout,
            color=color,
            fig_facecolor=fig_facecolor,
            axis_facecolor=axis_facecolor,
            axes=axes,
            block=block,
            show=show,
        )

    def get_peaks(
        self: SpectrumArray,
        smooth: bool = True,
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] = (1, 40),
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 6),
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

        peak_df = get_peak_params(
            self.get_data(),
            self.freqs,
            self.ch_names,
            smooth=smooth,
            smoothing_window=smoothing_window,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            polyorder=polyorder,
            peak_width_limits=peak_width_limits,
        )

        return peak_df


class AperiodicSpectrumArray(SpectrumArray):
    """Subclass of SpectrumArray"""

    def __init__(
        self: SpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ):
        # _check_data_shape(data, freqs, info, ndim=2)

        self.__setstate__(
            dict(
                method='IRASA',
                data=data,
                sfreq=info['sfreq'],
                dims=('channel', 'freq'),
                freqs=freqs,
                inst_type_str='Raw',
                data_type='Aperiodic Power Spectrum',
                info=info,
            )
        )

    def fit_aperiodic_model(
        self: SpectrumArray,
        fit_func: str = 'fixed',
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
            An object containing three pandas DataFrames:
                - aperiodic_params : pd.DataFrame
                    A DataFrame containing the fitted aperiodic parameters for each channel.
                - gof : pd.DataFrame
                    A DataFrame containing the goodness of fit metrics for each channel.
                - model : pd.DataFrame
                    A DataFrame containing the predicted aperiodic model for each channel and each time point.


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
            self.get_data(),
            self.freqs,
            ch_names=self.ch_names,
            scale=scale,
            fit_func=fit_func,
            fit_bounds=fit_bounds,
        )


# %%
class PeriodicEpochsSpectrum(EpochsSpectrumArray):
    """Subclass of EpochsSpectrumArray"""

    def __init__(
        self: EpochsSpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        events: np.ndarray | None = None,
        event_id: int | list | dict | str | None = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> None:
        if events is not None and data.shape[0] != events.shape[0]:
            raise ValueError(
                f'The first dimension of `data` ({data.shape[0]}) must match the '
                f'first dimension of `events` ({events.shape[0]}).'
            )

        self.__setstate__(
            dict(
                method='IRASA',
                data=data,
                sfreq=info['sfreq'],
                dims=('epoch', 'channel', 'freq'),
                freqs=freqs,
                inst_type_str='Epochs',
                data_type='Periodic Power Spectrum',
                info=info,
                events=events,
                event_id=event_id,
                metadata=None,
                selection=np.arange(data.shape[0]),
                drop_log=tuple(tuple() for _ in range(data.shape[0])),
            )
        )

    def plot(
        self: EpochsSpectrumArray,
        *,
        picks: str | np.ndarray | slice | None = None,
        average: bool = False,
        dB: bool = False,  # noqa N803
        amplitude: bool = False,
        xscale: str = 'linear',
        ci: float | str | None = 'sd',
        ci_alpha: float = 0.3,
        color: str | tuple = 'black',
        alpha: float | None = None,
        spatial_colors: bool = True,
        sphere: float | np.ndarray | mne.bem.ConductorModel | None | str = None,
        exclude: list | tuple | str = (),
        axes: matplotlib.axes.Axes | list | None = None,
        show: bool = True,
    ) -> None:
        super().plot(
            picks=picks,
            average=average,
            dB=dB,
            amplitude=amplitude,
            xscale=xscale,
            ci=ci,
            ci_alpha=ci_alpha,
            color=color,
            alpha=alpha,
            spatial_colors=spatial_colors,
            sphere=sphere,
            exclude=exclude,
            axes=axes,
            show=show,
        )

    def plot_topo(
        self: EpochsSpectrumArray,
        *,
        dB: bool = False,  # noqa N803
        layout: mne.channels.Layout | None = None,
        color: str | tuple = 'w',
        fig_facecolor: str | tuple = 'k',
        axis_facecolor: str | tuple = 'k',
        axes: matplotlib.axes.Axes | list | None = None,
        block: bool = False,
        show: bool = True,
    ) -> None:
        super().plot_topo(
            dB=dB,
            layout=layout,
            color=color,
            fig_facecolor=fig_facecolor,
            axis_facecolor=axis_facecolor,
            axes=axes,
            block=block,
            show=show,
        )

    def get_peaks(
        self: EpochsSpectrumArray,
        smooth: bool = True,
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] = (1.0, 40.0),
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 6.0),
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

        event_dict = {val: key for key, val in self.event_id.items()}
        events = self.events[:, 2]

        peak_list = []
        for ix, cur_epoch in enumerate(self.get_data()):
            peak_df = get_peak_params(
                cur_epoch,
                self.freqs,
                self.ch_names,
                smooth=smooth,
                smoothing_window=smoothing_window,
                cut_spectrum=cut_spectrum,
                peak_threshold=peak_threshold,
                min_peak_height=min_peak_height,
                polyorder=polyorder,
                peak_width_limits=peak_width_limits,
            )

            peak_df['event_id'] = event_dict[events[ix]]

            peak_list.append(peak_df)

        return pd.concat(peak_list)


class AperiodicEpochsSpectrum(EpochsSpectrumArray):
    """Subclass of EpochsSpectrumArray"""

    def __init__(
        self: EpochsSpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        events: np.ndarray | None = None,
        event_id: int | list | dict | str | None = None,
        *,
        verbose: bool | str | int | None = None,
    ):
        if events is not None and data.shape[0] != events.shape[0]:
            raise ValueError(
                f'The first dimension of `data` ({data.shape[0]}) must match the '
                f'first dimension of `events` ({events.shape[0]}).'
            )

        self.__setstate__(
            dict(
                method='IRASA',
                data=data,
                sfreq=info['sfreq'],
                dims=('epoch', 'channel', 'freq'),
                freqs=freqs,
                inst_type_str='Epochs',
                data_type='Aperiodic Power Spectrum',
                info=info,
                events=events,
                event_id=event_id,
                metadata=None,
                selection=np.arange(data.shape[0]),
                drop_log=tuple(tuple() for _ in range(data.shape[0])),
            )
        )

    def fit_aperiodic_model(
        self: SpectrumArray,
        fit_func: str = 'fixed',
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
            An object containing three pandas DataFrames:
                - aperiodic_params : pd.DataFrame
                    A DataFrame containing the fitted aperiodic parameters for each channel.
                - gof : pd.DataFrame
                    A DataFrame containing the goodness of fit metrics for each channel.
                - model : pd.DataFrame
                    A DataFrame containing the predicted aperiodic model for each channel and each time point.


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

        event_dict = {val: key for key, val in self.event_id.items()}
        events = self.events[:, 2]

        aps_list, gof_list, pred_list = [], [], []
        for ix, cur_epoch in enumerate(self.get_data()):
            slope_fit = compute_aperiodic_model(
                cur_epoch,
                self.freqs,
                ch_names=self.ch_names,
                scale=scale,
                fit_func=fit_func,
                fit_bounds=fit_bounds,
            )

            slope_fit.aperiodic_params['event_id'] = event_dict[events[ix]]
            slope_fit.gof['event_id'] = event_dict[events[ix]]
            aps_list.append(slope_fit.aperiodic_params.copy())
            gof_list.append(slope_fit.gof.copy())
            pred_list.append(slope_fit.model)

        return AperiodicFit(aperiodic_params=pd.concat(aps_list), gof=pd.concat(gof_list), model=pd.concat(pred_list))


@define
class IrasaRaw:
    periodic: PeriodicSpectrumArray
    aperiodic: AperiodicSpectrumArray


@define
class IrasaEpoched:
    periodic: PeriodicEpochsSpectrum
    aperiodic: AperiodicEpochsSpectrum
