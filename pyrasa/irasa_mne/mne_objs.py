"""Classes for the MNE Python interface."""

# %% inherit from spectrum array

from typing import Literal, Self

import matplotlib
import mne
import numpy as np
import pandas as pd
from attrs import define
from mne.time_frequency import EpochsSpectrumArray, SpectrumArray

from pyrasa.utils.aperiodic_utils import compute_aperiodic_model
from pyrasa.utils.fit_funcs import AbstractFitFun
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
        aperiodic_spectrum: np.ndarray,
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
        self._aperiodic_spectrum = aperiodic_spectrum

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
    """
    IRASA decomposition of continuous MNE data.

    Contains the periodic and aperiodic power spectra as MNE-compatible
    SpectrumArray objects, together with methods for extracting
    oscillatory peaks and fitting the aperiodic component.

    Attributes
    ----------
    periodic : PeriodicSpectrumArray
        Oscillatory component of the power spectrum.
    aperiodic : AperiodicSpectrumArray
        Aperiodic component of the power spectrum.
    """

    periodic: PeriodicSpectrumArray
    aperiodic: AperiodicSpectrumArray

    def __attrs_post_init__(self) -> None:
        """Verify spectral alignment after initialization."""
        self._check_alignment()

    def _check_alignment(self) -> None:
        """Check that both spectra describe the same channels and frequencies."""
        if self.periodic.ch_names != self.aperiodic.ch_names:
            raise ValueError('Periodic and aperiodic channels are not aligned.')

        if not np.array_equal(self.periodic.freqs, self.aperiodic.freqs):
            raise ValueError('Periodic and aperiodic frequencies are not aligned.')

    def get_peaks(
        self,
        *,
        smooth: bool = True,
        smoothing_window: int | float = 2,
        polyorder: int | None = None,
        cut_spectrum: tuple[float, float] | None = (1, 40),
        peak_threshold: float = 2.0,
        min_peak_height: float = 0.0,
        min_peak_distance_hz: float | None = None,
        peak_width_limits: tuple[float, float] = (1.0, 6.0),
    ) -> pd.DataFrame:
        """
        Extract oscillatory peaks from the IRASA decomposition.

        Peak detection is performed on the periodic spectrum, while
        the aperiodic spectrum is used to calculate relative peak power.

        Parameters
        ----------
        smooth : bool, optional
            Whether to smooth the periodic spectrum before peak detection.
            Default is True.
        smoothing_window : int or float, optional
            Savitzky-Golay smoothing window in Hz. Default is 2 Hz.
        polyorder : int or None, optional
            Polynomial order of the Savitzky-Golay filter. If None,
            the order is selected automatically.
        cut_spectrum : tuple of (float, float) or None, optional
            Frequency range in Hz used for peak detection.
            Default is (1, 40). If None, use the full spectrum.
        peak_threshold : float, optional
            Minimum peak prominence expressed as a multiple of the
            standard deviation of the processed periodic spectrum.
            Default is 2.0.
        min_peak_height : float, optional
            Minimum peak prominence in linear power units.
            Default is 0.0.
        min_peak_distance_hz : float or None, optional
            Minimum distance between detected peaks in Hz.
        peak_width_limits : tuple of (float, float), optional
            Minimum and maximum accepted peak widths in Hz.
            Default is (1.0, 6.0).

        Returns
        -------
        pd.DataFrame
            Detected peaks with the following columns:

            - ch_name : Channel name.
            - cf : Center frequency in Hz.
            - bw : Full width at half-prominence in Hz.
            - pw : Peak prominence in linear power units.
            - pw_rel : Log10 peak height relative to the aperiodic background.

        Notes
        -----
        Peak detection uses scipy.signal.find_peaks. If smoothing is
        enabled, relative peak power is calculated using the smoothed
        periodic spectrum and the original aperiodic background.
        """

        if self.periodic.ch_names != self.aperiodic.ch_names:
            raise ValueError('Periodic and aperiodic channels are not aligned.')

        if not np.array_equal(self.periodic.freqs, self.aperiodic.freqs):
            raise ValueError('Periodic and aperiodic frequencies are not aligned.')

        periodic = self.periodic.get_data(
            picks=self.periodic.ch_names,
            exclude=(),
        )
        aperiodic = self.aperiodic.get_data(
            picks=self.aperiodic.ch_names,
            exclude=(),
        )

        if periodic.shape != aperiodic.shape:
            raise ValueError('Periodic and aperiodic spectrum shapes do not match.')

        return get_peak_params(
            periodic_spectrum=periodic,
            aperiodic_spectrum=aperiodic,
            freqs=self.periodic.freqs,
            ch_names=self.periodic.ch_names,
            smooth=smooth,
            smoothing_window=smoothing_window,
            polyorder=polyorder,
            cut_spectrum=cut_spectrum,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            min_peak_distance_hz=min_peak_distance_hz,
            peak_width_limits=peak_width_limits,
        )

    def fit_aperiodic_model(
        self,
        fit_func: str | type[AbstractFitFun] = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> AperiodicFit:
        """
        Fit an aperiodic model to the IRASA aperiodic spectrum.

        Parameters
        ----------
        fit_func : str or type[AbstractFitFun], optional
            Model used for fitting. Either 'fixed', 'knee', or a custom
            subclass of AbstractFitFun. Default is 'fixed'.
        scale : bool, optional
            Whether to scale the power spectrum to improve numerical
            stability during fitting. Default is False.
        fit_bounds : tuple of (float, float) or None, optional
            Lower and upper frequency bounds in Hz. If None, the full
            frequency range is used.

        Returns
        -------
        AperiodicFit
            Fitting results containing the aperiodic parameters,
            goodness-of-fit metrics, and model predictions.
        """
        return self.aperiodic.fit_aperiodic_model(
            fit_func=fit_func,
            scale=scale,
            fit_bounds=fit_bounds,
        )

    def copy(self) -> Self:
        """
        Return an independent copy of the IRASA decomposition.

        Both the periodic and aperiodic MNE spectrum objects are copied.

        Returns
        -------
        IrasaRaw | IrasaEpoched
            A new IRASA result with independent spectral data.
        """
        return type(self)(
            periodic=self.periodic.copy(),
            aperiodic=self.aperiodic.copy(),
        )

    def pick(
        self,
        picks: str | list[str] | list[int] | slice | np.ndarray | None,
        exclude: str | list[str] | tuple[str, ...] = (),
    ) -> Self:
        """
        Select channels in both IRASA spectral components.

        The same MNE channel selection is applied to the periodic and
        aperiodic spectra. The operation modifies this object in place.

        Parameters
        ----------
        picks : str | list | slice | ndarray | None
            Channels to retain, following MNE's channel-selection
            conventions. Accepts channel names, indices, channel types
            (e.g., 'eeg' or 'meg'), or 'all'.
        exclude : str | list of str | tuple of str, optional
            Channels to exclude from type-based selections.
            Default is (), meaning no exclusions.

        Returns
        -------
        IrasaRaw | IrasaEpoched
            The modified IRASA result.
        """
        self._check_alignment()

        result = self.copy()
        result.periodic.pick(picks, exclude=exclude)
        result.aperiodic.pick(picks, exclude=exclude)
        result._check_alignment()

        self.periodic, self.aperiodic = result.periodic, result.aperiodic
        return self

    def drop_channels(
        self,
        ch_names: str | list[str],
        on_missing: Literal['raise', 'warn', 'ignore'] = 'raise',
    ) -> Self:
        """
        Remove channels from both IRASA spectral components.

        Parameters
        ----------
        ch_names : str | list of str
            Channel names to remove.
        on_missing : {'raise', 'warn', 'ignore'}
            Behavior when a requested channel does not exist.
            Default is 'raise'.

        Returns
        -------
        IrasaRaw | IrasaEpoched
            The modified IRASA result.
        """
        self._check_alignment()

        result = self.copy()
        result.periodic.drop_channels(ch_names, on_missing=on_missing)
        result.aperiodic.drop_channels(ch_names, on_missing=on_missing)
        result._check_alignment()

        self.periodic, self.aperiodic = result.periodic, result.aperiodic
        return self


@define
class IrasaEpoched:
    """
    IRASA decomposition of epoched MNE data.

    Contains the periodic and aperiodic power spectra as MNE-compatible
    SpectrumArray objects, together with methods for extracting
    oscillatory peaks and fitting the aperiodic component.

    Attributes
    ----------
    periodic : PeriodicSpectrumArray
        Oscillatory component of the power spectrum.
    aperiodic : AperiodicSpectrumArray
        Aperiodic component of the power spectrum.
    """

    periodic: PeriodicEpochsSpectrum
    aperiodic: AperiodicEpochsSpectrum

    def __getitem__(
        self,
        item: int | slice | list[int] | list[str] | np.ndarray | str,
    ) -> Self:
        """
        Select epochs from both IRASA spectral components.

        Supports the same epoch-selection syntax as MNE EpochsSpectrum,
        including integer indices, slices, lists of indices, boolean
        masks, and event names.

        Parameters
        ----------
        item : int | slice | list | ndarray | str
            Epoch selection. Event names must exist in the spectrum's
            event_id mapping.

        Returns
        -------
        IrasaEpoched
            A new IRASA result containing only the selected epochs.
            The original object remains unchanged.
        """
        self._check_alignment()

        result = type(self)(
            periodic=self.periodic[item],
            aperiodic=self.aperiodic[item],
        )
        result._check_alignment()
        return result

    def __attrs_post_init__(self) -> None:
        """Verify spectral alignment after initialization."""
        self._check_alignment()

    def _check_alignment(self) -> None:
        """Check channel, frequency, and epoch alignment."""
        if self.periodic.ch_names != self.aperiodic.ch_names:
            raise ValueError('Periodic and aperiodic channels are not aligned.')

        if not np.array_equal(self.periodic.freqs, self.aperiodic.freqs):
            raise ValueError('Periodic and aperiodic frequencies are not aligned.')

        if not np.array_equal(self.periodic.events, self.aperiodic.events):
            raise ValueError('Periodic and aperiodic epochs are not aligned.')

        if not np.array_equal(self.periodic.selection, self.aperiodic.selection):
            raise ValueError('Periodic and aperiodic epoch selections are not aligned.')

        if self.periodic.event_id != self.aperiodic.event_id:
            raise ValueError('Periodic and aperiodic event IDs are not aligned.')

    def get_peaks(
        self,
        *,
        smooth: bool = True,
        smoothing_window: int | float = 2,
        polyorder: int | None = None,
        cut_spectrum: tuple[float, float] | None = (1, 40),
        peak_threshold: float = 2.0,
        min_peak_height: float = 0.0,
        min_peak_distance_hz: float | None = None,
        peak_width_limits: tuple[float, float] = (1.0, 6.0),
    ) -> pd.DataFrame:
        """
        Extract oscillatory peak parameters for each epoch and channel of the IRASA decomposition.

        Peak detection is performed on the periodic spectrum, while
        the aperiodic spectrum is used to calculate relative peak power.

        Parameters
        ----------
        smooth : bool, optional
            Whether to smooth the periodic spectrum before peak detection.
            Default is True.
        smoothing_window : int or float, optional
            Savitzky-Golay smoothing window in Hz. Default is 2 Hz.
        polyorder : int or None, optional
            Polynomial order of the Savitzky-Golay filter. If None,
            the order is selected automatically.
        cut_spectrum : tuple of (float, float) or None, optional
            Frequency range in Hz used for peak detection.
            Default is (1, 40). If None, use the full spectrum.
        peak_threshold : float, optional
            Minimum peak prominence expressed as a multiple of the
            standard deviation of the processed periodic spectrum.
            Default is 2.0.
        min_peak_height : float, optional
            Minimum peak prominence in linear power units.
            Default is 0.0.
        min_peak_distance_hz : float or None, optional
            Minimum distance between detected peaks in Hz.
        peak_width_limits : tuple of (float, float), optional
            Minimum and maximum accepted peak widths in Hz.
            Default is (1.0, 6.0).

        Returns
        -------
        pd.DataFrame
            Detected peaks with the following columns:

            - ch_name : Channel name.
            - cf : Center frequency in Hz.
            - bw : Full width at half-prominence in Hz.
            - pw : Peak prominence in linear power units.
            - pw_rel : Log10 peak height relative to the aperiodic background.

        Notes
        -----
        Peak detection uses scipy.signal.find_peaks. If smoothing is
        enabled, relative peak power is calculated using the smoothed
        periodic spectrum and the original aperiodic background.
        """
        if self.periodic.ch_names != self.aperiodic.ch_names:
            raise ValueError('Periodic and aperiodic channels are not aligned.')

        if not np.array_equal(self.periodic.freqs, self.aperiodic.freqs):
            raise ValueError('Periodic and aperiodic frequencies are not aligned.')

        if not np.array_equal(self.periodic.events, self.aperiodic.events):
            raise ValueError('Periodic and aperiodic epochs are not aligned.')

        periodic = self.periodic.get_data(
            picks=self.periodic.ch_names,
            exclude=(),
        )
        aperiodic = self.aperiodic.get_data(
            picks=self.aperiodic.ch_names,
            exclude=(),
        )

        if periodic.shape != aperiodic.shape:
            raise ValueError('Periodic and aperiodic spectrum shapes do not match.')

        event_dict = {value: key for key, value in self.periodic.event_id.items()}
        peak_list = []

        for ix, (cur_periodic, cur_aperiodic) in enumerate(zip(periodic, aperiodic, strict=True)):
            peak_df = get_peak_params(
                periodic_spectrum=cur_periodic,
                aperiodic_spectrum=cur_aperiodic,
                freqs=self.periodic.freqs,
                ch_names=self.periodic.ch_names,
                smooth=smooth,
                smoothing_window=smoothing_window,
                polyorder=polyorder,
                cut_spectrum=cut_spectrum,
                peak_threshold=peak_threshold,
                min_peak_height=min_peak_height,
                min_peak_distance_hz=min_peak_distance_hz,
                peak_width_limits=peak_width_limits,
            )

            peak_df['event_id'] = event_dict[self.periodic.events[ix, 2]]
            peak_list.append(peak_df)

        return pd.concat(peak_list, ignore_index=True)

    def fit_aperiodic_model(
        self,
        fit_func: str | type[AbstractFitFun] = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> AperiodicFit:
        """Fit an aperiodic model separately to each epoch and channel of the IRASA aperiodic spectrum.

        Parameters
        ----------
        fit_func : str or type[AbstractFitFun], optional
            Model used for fitting. Either 'fixed', 'knee', or a custom
            subclass of AbstractFitFun. Default is 'fixed'.
        scale : bool, optional
            Whether to scale the power spectrum to improve numerical
            stability during fitting. Default is False.
        fit_bounds : tuple of (float, float) or None, optional
            Lower and upper frequency bounds in Hz. If None, the full
            frequency range is used.

        Returns
        -------
        AperiodicFit
            Fitting results containing the aperiodic parameters,
            goodness-of-fit metrics, and model predictions.
        """
        return self.aperiodic.fit_aperiodic_model(
            fit_func=fit_func,
            scale=scale,
            fit_bounds=fit_bounds,
        )

    def copy(self) -> Self:
        """
        Return an independent copy of the IRASA decomposition.

        Both the periodic and aperiodic MNE spectrum objects are copied.

        Returns
        -------
        IrasaRaw | IrasaEpoched
            A new IRASA result with independent spectral data.
        """
        return type(self)(
            periodic=self.periodic.copy(),
            aperiodic=self.aperiodic.copy(),
        )

    def pick(
        self,
        picks: str | list[str] | list[int] | slice | np.ndarray | None,
        exclude: str | list[str] | tuple[str, ...] = (),
    ) -> Self:
        """
        Select channels in both IRASA spectral components.

        The same MNE channel selection is applied to the periodic and
        aperiodic spectra. The operation modifies this object in place.

        Parameters
        ----------
        picks : str | list | slice | ndarray | None
            Channels to retain, following MNE's channel-selection
            conventions. Accepts channel names, indices, channel types
            (e.g., 'eeg' or 'meg'), or 'all'.
        exclude : str | list of str | tuple of str, optional
            Channels to exclude from type-based selections.
            Default is (), meaning no exclusions.

        Returns
        -------
        IrasaRaw | IrasaEpoched
            The modified IRASA result.
        """
        self._check_alignment()

        result = self.copy()
        result.periodic.pick(picks, exclude=exclude)
        result.aperiodic.pick(picks, exclude=exclude)
        result._check_alignment()

        self.periodic, self.aperiodic = result.periodic, result.aperiodic
        return self

    def drop_channels(
        self,
        ch_names: str | list[str],
        on_missing: Literal['raise', 'warn', 'ignore'] = 'raise',
    ) -> Self:
        """
        Remove channels from both IRASA spectral components.

        Parameters
        ----------
        ch_names : str | list of str
            Channel names to remove.
        on_missing : {'raise', 'warn', 'ignore'}
            Behavior when a requested channel does not exist.
            Default is 'raise'.

        Returns
        -------
        IrasaRaw | IrasaEpoched
            The modified IRASA result.
        """
        self._check_alignment()

        result = self.copy()
        result.periodic.drop_channels(ch_names, on_missing=on_missing)
        result.aperiodic.drop_channels(ch_names, on_missing=on_missing)
        result._check_alignment()

        self.periodic, self.aperiodic = result.periodic, result.aperiodic
        return self
