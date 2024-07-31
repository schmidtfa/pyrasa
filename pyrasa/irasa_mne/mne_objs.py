"""Classes for the MNE Python interface."""

import matplotlib
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import EpochsSpectrumArray, SpectrumArray

from pyrasa.utils.aperiodic_utils import compute_slope
from pyrasa.utils.peak_utils import get_peak_params


class PeriodicSpectrumArray(SpectrumArray):
    """Subclass of SpectrumArray."""

    def __init__(
        self: SpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ) -> None:
        """
        Initialize the PeriodicSpectrumArray.

        Parameters
        ----------
        data : np.ndarray
            The data array.
        info : mne.Info
            The info object.
        freqs : np.ndarray
            The frequencies array.
        verbose : bool | str | int | None, optional
            The verbosity level.
        """
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
        """
        Plot the spectrum.

        Parameters
        ----------
        picks : str | np.ndarray | slice | None, optional
            Channels to include. If None, all available channels are used.
        average : bool, optional
            Whether to average the data. Defaults to False.
        dB : bool, optional
            If True, convert data to decibels (dB). Defaults to False.
        amplitude : bool | str, optional
            If True, plot amplitude spectrum. Defaults to False.
        xscale : str, optional
            Scale of the x-axis. Defaults to 'linear'.
        ci : float | str | None, optional
            Confidence interval. Defaults to 'sd'.
        ci_alpha : float, optional
            Alpha value for the confidence interval. Defaults to 0.3.
        color : str | tuple, optional
            Color of the plot. Defaults to 'black'.
        alpha : float | None, optional
            Alpha value for the plot. Defaults to None.
        spatial_colors : bool, optional
            Use spatial colors for the plot. Defaults to True.
        sphere : float | np.ndarray | mne.bem.ConductorModel | None | str, optional
            Sphere parameters. Defaults to None.
        exclude : tuple | list | str, optional
            Channels to exclude. Defaults to ().
        axes : matplotlib.axes.Axes | list | None, optional
            Axes to plot on. Defaults to None.
        show : bool, optional
            Whether to show the plot. Defaults to True.
        """
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
        """
        Plot the topography.

        Parameters
        ----------
        dB : bool, optional
            If True, convert data to decibels (dB). Defaults to False.
        layout : mne.channels.Layout | None, optional
            Layout of the topography. Defaults to None.
        color : str | tuple, optional
            Color of the plot. Defaults to 'w'.
        fig_facecolor : str | tuple, optional
            Face color of the figure. Defaults to 'k'.
        axis_facecolor : str | tuple, optional
            Face color of the axis. Defaults to 'k'.
        axes : matplotlib.axes.Axes | list | None, optional
            Axes to plot on. Defaults to None.
        block : bool, optional
            Whether to block the plot. Defaults to False.
        show : bool, optional
            Whether to show the plot. Defaults to True.
        """
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
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] = (1, 40),
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 6),
    ) -> pd.DataFrame:
        """
        Extract peak parameters from the periodic spectrum.

        The algorithm works by smoothing the spectrum, zeroing out negative values,
        and extracting peaks based on user-specified parameters.

        Parameters
        ----------
        smoothing_window : float | int, optional
            Smoothing window in Hz for the Savitzky-Golay filter. Defaults to 1.
        cut_spectrum : tuple of float, optional
            Frequency range for peak detection. Defaults to (1, 40).
        peak_threshold : float, optional
            Relative threshold for detecting peaks. Defaults to 2.5.
        min_peak_height : float, optional
            Absolute threshold for identifying peaks. Defaults to 0.0.
        polyorder : int, optional
            Polynomial order for the Savitzky-Golay filter. Defaults to 1.
        peak_width_limits : tuple of float, optional
            Limits on possible peak width in Hz. Defaults to (0.5, 6).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the center frequency, bandwidth, and peak height for each channel.
        """
        peak_df = get_peak_params(
            self.get_data(),
            self.freqs,
            self.ch_names,
            smoothing_window=smoothing_window,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            polyorder=polyorder,
            peak_width_limits=peak_width_limits,
        )

        return peak_df


class AperiodicSpectrumArray(SpectrumArray):
    """Subclass of SpectrumArray."""

    def __init__(
        self: SpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        *,
        verbose: bool | str | int | None = None,
    ) -> None:
        """
        Initialize the AperiodicSpectrumArray.

        Parameters
        ----------
        data : np.ndarray
            The data array.
        info : mne.Info
            The info object.
        freqs : np.ndarray
            The frequencies array.
        verbose : bool | str | int | None, optional
            The verbosity level.
        """
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

    def get_slopes(
        self: SpectrumArray,
        fit_func: str = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract aperiodic parameters from the aperiodic spectrum.

        The algorithm applies one of two different curve fit functions
        and returns the associated parameters and goodness of fit.

        Parameters
        ----------
        fit_func : str, optional
            Fit function to use ('fixed' or 'knee'). Defaults to 'fixed'.
        scale : bool, optional
            Whether to scale the data. Defaults to False.
        fit_bounds : tuple of float | None, optional
            Lower and upper bounds for the fit function. Defaults to None.

        Returns
        -------
        tuple of pd.DataFrame
            DataFrame containing the aperiodic parameters for each channel.
            DataFrame containing the goodness of fit for each channel.
        """
        df_aps, df_gof = compute_slope(
            self.get_data(),
            self.freqs,
            ch_names=self.ch_names,
            scale=scale,
            fit_func=fit_func,
            fit_bounds=fit_bounds,
        )

        return df_aps, df_gof


# %%
class PeriodicEpochsSpectrum(EpochsSpectrumArray):
    """Subclass of EpochsSpectrumArray."""

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
        """
        Initialize the PeriodicEpochsSpectrum.

        Parameters
        ----------
        data : np.ndarray
            The data array.
        info : mne.Info
            The info object.
        freqs : np.ndarray
            The frequencies array.
        events : np.ndarray | None, optional
            The events array. Defaults to None.
        event_id : int | list | dict | str | None, optional
            The event ID. Defaults to None.
        verbose : bool | str | int | None, optional
            The verbosity level.
        """
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
        """
        Plot the epochs spectrum.

        Parameters
        ----------
        picks : str | np.ndarray | slice | None, optional
            Channels to include. If None, all available channels are used.
        average : bool, optional
            Whether to average the data. Defaults to False.
        dB : bool, optional
            If True, convert data to decibels (dB). Defaults to False.
        amplitude : bool, optional
            If True, plot amplitude spectrum. Defaults to False.
        xscale : str, optional
            Scale of the x-axis. Defaults to 'linear'.
        ci : float | str | None, optional
            Confidence interval. Defaults to 'sd'.
        ci_alpha : float, optional
            Alpha value for the confidence interval. Defaults to 0.3.
        color : str | tuple, optional
            Color of the plot. Defaults to 'black'.
        alpha : float | None, optional
            Alpha value for the plot. Defaults to None.
        spatial_colors : bool, optional
            Use spatial colors for the plot. Defaults to True.
        sphere : float | np.ndarray | mne.bem.ConductorModel | None | str, optional
            Sphere parameters. Defaults to None.
        exclude : list | tuple | str, optional
            Channels to exclude. Defaults to ().
        axes : matplotlib.axes.Axes | list | None, optional
            Axes to plot on. Defaults to None.
        show : bool, optional
            Whether to show the plot. Defaults to True.
        """
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
        """
        Plot the topography of the epochs spectrum.

        Parameters
        ----------
        dB : bool, optional
            If True, convert data to decibels (dB). Defaults to False.
        layout : mne.channels.Layout | None, optional
            Layout of the topography. Defaults to None.
        color : str | tuple, optional
            Color of the plot. Defaults to 'w'.
        fig_facecolor : str | tuple, optional
            Face color of the figure. Defaults to 'k'.
        axis_facecolor : str | tuple, optional
            Face color of the axis. Defaults to 'k'.
        axes : matplotlib.axes.Axes | list | None, optional
            Axes to plot on. Defaults to None.
        block : bool, optional
            Whether to block the plot. Defaults to False.
        show : bool, optional
            Whether to show the plot. Defaults to True.
        """
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
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] = (1.0, 40.0),
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 6.0),
    ) -> pd.DataFrame:
        """
        Extract peak parameters from the periodic spectrum.

        The algorithm works by smoothing the spectrum, zeroing out negative values,
        and extracting peaks based on user-specified parameters.

        Parameters
        ----------
        smoothing_window : float | int, optional
            Smoothing window in Hz for the Savitzky-Golay filter. Defaults to 1.
        cut_spectrum : tuple of float, optional
            Frequency range for peak detection. Defaults to (1.0, 40.0).
        peak_threshold : float, optional
            Relative threshold for detecting peaks. Defaults to 2.5.
        min_peak_height : float, optional
            Absolute threshold for identifying peaks. Defaults to 0.0.
        polyorder : int, optional
            Polynomial order for the Savitzky-Golay filter. Defaults to 1.
        peak_width_limits : tuple of float, optional
            Limits on possible peak width in Hz. Defaults to (0.5, 6.0).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the center frequency, bandwidth, and peak height for each channel.
        """
        event_dict = {val: key for key, val in self.event_id.items()}
        events = self.events[:, 2]

        peak_list = []
        for ix, cur_epoch in enumerate(self.get_data()):
            peak_df = get_peak_params(
                cur_epoch,
                self.freqs,
                self.ch_names,
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
    """Subclass of EpochsSpectrumArray."""

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
        """
        Initialize the AperiodicEpochsSpectrum.

        Parameters
        ----------
        data : np.ndarray
            The data array.
        info : mne.Info
            The info object.
        freqs : np.ndarray
            The frequencies array.
        events : np.ndarray | None, optional
            The events array. Defaults to None.
        event_id : int | list | dict | str | None, optional
            The event ID. Defaults to None.
        verbose : bool | str | int | None, optional
            The verbosity level.
        """
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

    def get_slopes(
        self: SpectrumArray,
        fit_func: str = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract aperiodic parameters from the aperiodic spectrum.

        The algorithm applies one of two different curve fit functions
        and returns the associated parameters and goodness of fit.

        Parameters
        ----------
        fit_func : str, optional
            Fit function to use ('fixed' or 'knee'). Defaults to 'fixed'.
        scale : bool, optional
            Whether to scale the data. Defaults to False.
        fit_bounds : tuple of float | None, optional
            Lower and upper bounds for the fit function. Defaults to None.

        Returns
        -------
        tuple of pd.DataFrame
            DataFrame containing the aperiodic parameters for each channel.
            DataFrame containing the goodness of fit for each channel.
        """
        event_dict = {val: key for key, val in self.event_id.items()}
        events = self.events[:, 2]

        aps_list, gof_list = [], []
        for ix, cur_epoch in enumerate(self.get_data()):
            df_aps, df_gof = compute_slope(
                cur_epoch,
                self.freqs,
                ch_names=self.ch_names,
                scale=scale,
                fit_func=fit_func,
                fit_bounds=fit_bounds,
            )

            df_aps['event_id'] = event_dict[events[ix]]
            df_gof['event_id'] = event_dict[events[ix]]
            aps_list.append(df_aps)
            gof_list.append(df_gof)

        return pd.concat(aps_list), pd.concat(gof_list)
