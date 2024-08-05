# %% inherit from spectrum array

import matplotlib
import mne
import numpy as np
import pandas as pd
from attrs import define
from mne.time_frequency import EpochsSpectrumArray, SpectrumArray

from pyrasa.utils.aperiodic_utils import compute_slope
from pyrasa.utils.peak_utils import get_peak_params
from pyrasa.utils.types import SlopeFit

# FutureWarning:


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
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] = (1, 40),
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 6),
    ) -> pd.DataFrame:
        """
        This method can be used to extract peak parameters from the periodic spectrum extracted from IRASA.
        The algorithm works by smoothing the spectrum, zeroing out negative values and
        extracting peaks based on user specified parameters.

        Parameters: smoothing window : int, optional, default: 2
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

    def get_slopes(
        self: SpectrumArray,
        fit_func: str = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> SlopeFit:
        """
        This method can be used to extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.
        The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
        as well as the respective goodness of fit.

        Parameters:
                    fit_func : string
                        Can be either "fixed" or "knee".
                    fit_bounds : None, tuple
                        Lower and upper bound for the fit function,
                        should be None if the whole frequency range is desired.
                        Otherwise a tuple of (lower, upper)

        Returns:    df_aps: DataFrame
                        DataFrame containing the center frequency, bandwidth and peak height for each channel
                    df_gof: DataFrame
                        DataFrame containing the goodness of fit of the specific fit function for each channel.

        """

        return compute_slope(
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
        smoothing_window: float | int = 1,
        cut_spectrum: tuple[float, float] = (1.0, 40.0),
        peak_threshold: float = 2.5,
        min_peak_height: float = 0.0,
        polyorder: int = 1,
        peak_width_limits: tuple[float, float] = (0.5, 6.0),
    ) -> pd.DataFrame:
        """
        This method can be used to extract peak parameters from the periodic spectrum extracted from IRASA.
        The algorithm works by smoothing the spectrum, zeroing out negative values and
        extracting peaks based on user specified parameters.

        Parameters: smoothing window : int, optional, default: 2
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

    def get_slopes(
        self: SpectrumArray,
        fit_func: str = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> SlopeFit:
        """
        This method can be used to extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.
        The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
        as well as the respective goodness of fit.

        Parameters:
                    fit_func : string
                        Can be either "fixed" or "knee".
                    fit_bounds : None, tuple
                        Lower and upper bound for the fit function,
                        should be None if the whole frequency range is desired.
                        Otherwise a tuple of (lower, upper)

        Returns:    df_aps: DataFrame
                        DataFrame containing the center frequency, bandwidth and peak height for each channel
                    df_gof: DataFrame
                        DataFrame containing the goodness of fit of the specific fit function for each channel.

        """

        event_dict = {val: key for key, val in self.event_id.items()}
        events = self.events[:, 2]

        aps_list, gof_list = [], []
        for ix, cur_epoch in enumerate(self.get_data()):
            slope_fit = compute_slope(
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

        return SlopeFit(aperiodic_params=pd.concat(aps_list), gof=pd.concat(gof_list))


@define
class IrasaRaw:
    periodic: PeriodicSpectrumArray
    aperiodic: AperiodicSpectrumArray


@define
class IrasaEpoched:
    periodic: PeriodicEpochsSpectrum
    aperiodic: AperiodicEpochsSpectrum
