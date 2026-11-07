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
    """
    Represent the periodic component of the power spectrum estimated by IRASA.

    This class extends :class:`mne.time_frequency.SpectrumArray` to provide
    an MNE-compatible representation of the periodic (oscillatory) component
    of the power spectrum. The periodic component is obtained by subtracting
    the IRASA-estimated aperiodic spectrum from the original power spectrum.

    The class inherits MNE's methods for accessing, selecting, and
    visualizing spectral data.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_freqs)
        Periodic component of the power spectral density for each channel.
    info : mne.Info
        Measurement information containing channel names, channel types,
        sampling frequency, and sensor locations. The channel order must
        correspond to the first dimension of `data`.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the spectral estimates, in Hz.

    Notes
    -----
    Since the periodic component is calculated by subtraction, it can
    contain negative values. Therefore, the plotting methods default to
    `dB=False` to avoid applying a logarithmic transformation to
    non-positive spectral values.

    See Also
    --------
    mne.time_frequency.SpectrumArray
    AperiodicSpectrumArray
    """

    def __init__(
        self: SpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
    ) -> None:
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
        Plot the IRASA-estimated periodic component of the power spectrum.

        Visualize the periodic spectrum across frequencies for selected channels.
        The plotting functionality is inherited from
        :meth:`mne.time_frequency.SpectrumArray.plot`.

        Unlike the parent method, the default is `dB=False`, since the periodic
        spectrum can contain negative values.

        Parameters
        ----------
        picks : str | np.ndarray | slice | None, optional
            Channels to include in the plot. Can be channel names, channel types,
            channel indices, or a slice. If None, all data channels are included.
        average : bool, default False
            Whether to average the spectra across the selected channels before
            plotting.
        dB : bool, default False
            Whether to convert spectral values to decibels. This requires
            positive spectral values and is generally not recommended for
            IRASA-derived periodic spectra.
        amplitude : bool, default False
            Whether to display amplitude rather than power spectral values.
            This transformation is not meaningful for negative spectral values.
        xscale : {'linear', 'log'}, default 'linear'
            Scaling of the frequency axis.
        ci : float | {'sd', 'range'} | None, default 'sd'
            Type of confidence band to display when `average=True`.
            'sd' displays one standard deviation around the mean, 'range'
            displays the minimum and maximum values, and a float specifies
            a bootstrap confidence level in percent. If None, no band is shown.
        ci_alpha : float, default 0.3
            Transparency of the confidence band, between 0 and 1.
        color : str | tuple, default 'black'
            Matplotlib-compatible color for the spectral curves. Ignored when
            `spatial_colors=True`.
        alpha : float | None, optional
            Transparency of the spectral curves. If None, MNE determines
            the transparency based on the `average` parameter.
        spatial_colors : bool, default True
            Whether to color the spectral curves according to sensor location.
        sphere : float | np.ndarray | mne.bem.ConductorModel | str | None, optional
            Sphere definition used for the head outline in interactive
            topographical visualizations.
        exclude : tuple | list | str, default ()
            Channel names to exclude from the plot. Use 'bads' to exclude
            channels marked as bad in `info`.
        axes : matplotlib.axes.Axes | list | None, optional
            Matplotlib axes on which to draw the spectra. If None, new axes
            are created.
        show : bool, default True
            Whether to display the figure.

        See Also
        --------
        mne.time_frequency.SpectrumArray.plot
        PeriodicSpectrumArray.plot_topo
        """
        return super().plot(
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
        Plot periodic spectra arranged according to sensor locations.

        Visualize the IRASA-estimated periodic component of the power spectrum
        for each channel in a topographical layout. The plotting functionality
        is inherited from :meth:`mne.time_frequency.SpectrumArray.plot_topo`.

        The default is `dB=False` because the periodic spectrum can contain
        negative values.

        Parameters
        ----------
        dB : bool, default False
            Whether to convert spectral values to decibels. This requires
            positive spectral values and is generally not recommended for
            IRASA-derived periodic spectra.
        layout : mne.channels.Layout | None, optional
            Sensor layout used to arrange the spectra. If None, the layout
            is inferred from the measurement information when possible.
        color : str | tuple, default 'w'
            Matplotlib-compatible color for the spectral curves.
        fig_facecolor : str | tuple, default 'k'
            Background color of the figure.
        axis_facecolor : str | tuple, default 'k'
            Background color of the plotting axes.
        axes : matplotlib.axes.Axes | list | None, optional
            Matplotlib axes on which to draw the spectra. If None, new axes
            are created.
        block : bool, default False
            Whether to block program execution until the figure is closed.
            Behavior depends on the installed MNE version.
        show : bool, default True
            Whether to display the figure.

        See Also
        --------
        mne.time_frequency.SpectrumArray.plot_topo
        PeriodicSpectrumArray.plot
        """
        return super().plot_topo(
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
    """
    Represent the aperiodic component of the power spectrum estimated by IRASA.

    This class extends :class:`mne.time_frequency.SpectrumArray` to provide
    an MNE-compatible representation of the aperiodic (fractal) component
    of the power spectrum estimated using IRASA.

    In addition to the functionality inherited from MNE, the class provides
    a method to fit parametric models to the aperiodic spectrum and extract
    parameters describing its spectral shape.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_freqs)
        IRASA-estimated aperiodic component of the power spectral density
        for each channel.
    info : mne.Info
        Measurement information containing channel names, channel types,
        sampling frequency, and sensor locations. The channel order must
        correspond to the first dimension of `data`.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the spectral estimates, in Hz.
    verbose : bool | str | int | None, optional
        Accepted for compatibility with MNE's interface but currently unused.

    See Also
    --------
    mne.time_frequency.SpectrumArray
    PeriodicSpectrumArray
    AperiodicSpectrumArray.fit_aperiodic_model
    """

    def __init__(
        self: SpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
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
        Fit a parametric model to the IRASA-estimated aperiodic spectrum.

        Fit the aperiodic component of the power spectrum separately for each
        channel using scipy's nonlinear least-squares optimization.

        Two models are available by default: a fixed model describing a power-law
        relationship without a knee, and a knee model that additionally captures
        a bend in the aperiodic spectrum. Custom fitting functions can be
        provided as subclasses of :class:`~pyrasa.utils.fit_funcs.AbstractFitFun`.

        Parameters
        ----------
        fit_func : str | type[AbstractFitFun], default 'fixed'
            Model used to fit the aperiodic spectrum. Options are:

            - 'fixed': Fit a power-law model without a knee, characterized by
            an offset and exponent.
            - 'knee': Fit a model including a spectral knee, characterized by
            an offset, knee parameter, and two exponents.
            - A subclass of `AbstractFitFun` defining a custom fitting model.

        scale : bool, default False
            Whether to scale the aperiodic spectrum before fitting to improve
            numerical stability. This can be useful when power values are very
            small. The fitted parameters and model predictions are adjusted
            to the original data scale.
        fit_bounds : tuple[float, float] | None, optional
            Lower and upper frequency bounds for model fitting, in Hz.
            If None, the full available frequency range is used.
            Otherwise, only frequencies within the specified bounds are
            included in the fit.

        Returns
        -------
        AperiodicFit
            Object containing three pandas DataFrames:

            - `aperiodic_params`: Fitted model parameters for each channel.
            - `gof`: Goodness-of-fit metrics for each channel, including
            mean squared error, R-squared, AIC, and BIC.
            - `model`: Predicted aperiodic power values for each channel
            and frequency in the fitted range.

        Notes
        -----
        The built-in fitting models operate on logarithmically transformed
        power values and require positive spectral values.

        If the frequency range includes 0 Hz, this frequency is excluded
        from the fitting procedure.

        See Also
        --------
        pyrasa.utils.aperiodic_utils.compute_aperiodic_model
        pyrasa.utils.fit_funcs.AbstractFitFun
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
    """
    Represent the periodic component of epoched power spectra estimated by IRASA.

    This class extends :class:`mne.time_frequency.EpochsSpectrumArray` to
    provide an MNE-compatible representation of the periodic (oscillatory)
    component of the power spectrum for individual epochs.

    The periodic component is obtained by subtracting the IRASA-estimated
    aperiodic spectrum from the original power spectrum. Spectral estimates
    are stored separately for each epoch and channel, together with
    information about the corresponding experimental events.

    Parameters
    ----------
    data : np.ndarray, shape (n_epochs, n_channels, n_freqs)
        Periodic component of the power spectral density for each epoch
        and channel.
    info : mne.Info
        Measurement information containing channel names, channel types,
        sampling frequency, and sensor locations. The channel order must
        correspond to the second dimension of `data`.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the spectral estimates, in Hz.
    events : np.ndarray, shape (n_epochs, 3) | None, optional
        MNE-compatible event array containing the sample index, previous
        event value, and event identifier for each epoch. The number of
        events must match the number of epochs. If None, no event array
        is explicitly provided.
    event_id : int | list | dict | str | None, optional
        Event identifiers associated with the epochs. Can be used to
        identify or select experimental conditions following MNE's
        event conventions. If None, all available event identifiers
        are considered.

    Notes
    -----
    Since the periodic component is calculated by subtraction, it can
    contain negative values. Therefore, the plotting methods default to
    `dB=False` to avoid applying a logarithmic transformation to
    non-positive spectral values.

    Raises
    ------
    ValueError
        If the number of epochs in `data` does not match the number
        of rows in `events`.

    See Also
    --------
    mne.time_frequency.EpochsSpectrumArray
    PeriodicSpectrumArray
    AperiodicEpochsSpectrum
    """

    def __init__(
        self: EpochsSpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        events: np.ndarray | None = None,
        event_id: int | list | dict | str | None = None,
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
        """
        Plot the IRASA-estimated periodic component of epoched power spectra.

        Visualize the periodic spectra across frequencies for selected channels.
        The plotting functionality is inherited from
        :meth:`mne.time_frequency.EpochsSpectrumArray.plot`.

        Unlike the parent method, the default is `dB=False`, since the periodic
        spectrum can contain negative values.

        Parameters
        ----------
        picks : str | np.ndarray | slice | None, optional
            Channels to include in the plot. Can be channel names, channel types,
            channel indices, or a slice. If None, all data channels are included.
        average : bool, default False
            Whether to average the spectra across selected channels before
            plotting. This parameter does not control averaging across epochs.
        dB : bool, default False
            Whether to convert spectral values to decibels. This requires
            positive spectral values and is generally not recommended for
            IRASA-derived periodic spectra.
        amplitude : bool, default False
            Whether to display amplitude rather than power spectral values.
            This transformation is not meaningful for negative spectral values.
        xscale : {'linear', 'log'}, default 'linear'
            Scaling of the frequency axis.
        ci : float | {'sd', 'range'} | None, default 'sd'
            Type of confidence band to display when `average=True`.
            'sd' displays one standard deviation around the mean, 'range'
            displays the minimum and maximum values, and a float specifies
            a bootstrap confidence level in percent. If None, no band is shown.
        ci_alpha : float, default 0.3
            Transparency of the confidence band, between 0 and 1.
        color : str | tuple, default 'black'
            Matplotlib-compatible color for the spectral curves. Ignored when
            `spatial_colors=True`.
        alpha : float | None, optional
            Transparency of the spectral curves. If None, MNE determines
            the transparency based on the `average` parameter.
        spatial_colors : bool, default True
            Whether to color the spectral curves according to sensor location.
        sphere : float | np.ndarray | mne.bem.ConductorModel | str | None, optional
            Sphere definition used for the head outline in interactive
            topographical visualizations.
        exclude : list | tuple | str, default ()
            Channel names to exclude from the plot. Use 'bads' to exclude
            channels marked as bad in `info`.
        axes : matplotlib.axes.Axes | list | None, optional
            Matplotlib axes on which to draw the spectra. If None, new axes
            are created.
        show : bool, default True
            Whether to display the figure.

        See Also
        --------
        mne.time_frequency.EpochsSpectrumArray.plot
        PeriodicEpochsSpectrum.plot_topo
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
        Plot periodic spectra from epoched data arranged by sensor location.

        Visualize the IRASA-estimated periodic component of the power spectrum
        for each channel in a topographical layout. The plotting functionality
        is inherited from
        :meth:`mne.time_frequency.EpochsSpectrumArray.plot_topo`.

        The default is `dB=False` because the periodic spectrum can contain
        negative values.

        Parameters
        ----------
        dB : bool, default False
            Whether to convert spectral values to decibels. This requires
            positive spectral values and is generally not recommended for
            IRASA-derived periodic spectra.
        layout : mne.channels.Layout | None, optional
            Sensor layout used to arrange the spectra. If None, the layout
            is inferred from the measurement information when possible.
        color : str | tuple, default 'w'
            Matplotlib-compatible color for the spectral curves.
        fig_facecolor : str | tuple, default 'k'
            Background color of the figure.
        axis_facecolor : str | tuple, default 'k'
            Background color of the plotting axes.
        axes : matplotlib.axes.Axes | list | None, optional
            Matplotlib axes on which to draw the spectra. If None, new axes
            are created.
        block : bool, default False
            Whether to block program execution until the figure is closed.
            Behavior depends on the installed MNE version.
        show : bool, default True
            Whether to display the figure.

        See Also
        --------
        mne.time_frequency.EpochsSpectrumArray.plot_topo
        PeriodicEpochsSpectrum.plot
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


class AperiodicEpochsSpectrum(EpochsSpectrumArray):
    """
    Represent the aperiodic component of epoched power spectra estimated by IRASA.

    This class extends :class:`mne.time_frequency.EpochsSpectrumArray` to
    provide an MNE-compatible representation of the aperiodic (fractal)
    component of the power spectrum for individual epochs.

    Spectral estimates are stored separately for each epoch and channel,
    together with information about the corresponding experimental events.
    The class additionally provides a method to fit parametric models to
    the aperiodic spectrum of each epoch and channel.

    Parameters
    ----------
    data : np.ndarray, shape (n_epochs, n_channels, n_freqs)
        IRASA-estimated aperiodic component of the power spectral density
        for each epoch and channel.
    info : mne.Info
        Measurement information containing channel names, channel types,
        sampling frequency, and sensor locations. The channel order must
        correspond to the second dimension of `data`.
    freqs : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the spectral estimates, in Hz.
    events : np.ndarray, shape (n_epochs, 3) | None, optional
        MNE-compatible event array containing the sample index, previous
        event value, and event identifier for each epoch. The number of
        events must match the number of epochs.
    event_id : int | list | dict | str | None, optional
        Event identifiers associated with the epochs, following MNE's
        event conventions.

    Raises
    ------
    ValueError
        If the number of epochs in `data` does not match the number
        of rows in `events`.

    See Also
    --------
    mne.time_frequency.EpochsSpectrumArray
    AperiodicSpectrumArray
    PeriodicEpochsSpectrum
    AperiodicEpochsSpectrum.fit_aperiodic_model
    """

    def __init__(
        self: EpochsSpectrumArray,
        data: np.ndarray,
        info: mne.Info,
        freqs: np.ndarray,
        events: np.ndarray | None = None,
        event_id: int | list | dict | str | None = None,
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
        Fit a parametric model to the aperiodic spectrum of each epoch.

        Fit the IRASA-estimated aperiodic component of the power spectrum
        separately for each epoch and channel using scipy's nonlinear
        least-squares optimization.

        Two models are available by default: a fixed model describing a
        power-law relationship without a knee, and a knee model that
        additionally captures a bend in the aperiodic spectrum. Custom
        fitting functions can be provided as subclasses of `AbstractFitFun`.

        The fitted parameters and goodness-of-fit metrics are combined across
        epochs and annotated with the corresponding experimental event labels.

        Parameters
        ----------
        fit_func : str | type[AbstractFitFun], default 'fixed'
            Model used to fit the aperiodic spectrum. Options are:

            - 'fixed': Fit a power-law model without a knee, characterized
            by an offset and exponent.
            - 'knee': Fit a model including a spectral knee.
            - A subclass of `AbstractFitFun` defining a custom fitting model.

        scale : bool, default False
            Whether to scale the aperiodic spectrum before fitting to improve
            numerical stability. This can be useful when power values are very
            small. The fitted parameters and model predictions are adjusted
            to the original data scale.
        fit_bounds : tuple[float, float] | None, optional
            Lower and upper frequency bounds for model fitting, in Hz.
            If None, the full available frequency range is used.
            Otherwise, only frequencies within the specified bounds are
            included in the fit.

        Returns
        -------
        AperiodicFit
            Object containing three pandas DataFrames:

            - `aperiodic_params`: Fitted model parameters for each channel
            and epoch, including an `event_id` column with the corresponding
            experimental event label.
            - `gof`: Goodness-of-fit metrics for each channel and epoch,
            including an `event_id` column.
            - `model`: Predicted aperiodic power values for each channel
            and epoch at the fitted frequencies.

        Notes
        -----
        Models are fitted independently for each epoch and channel.
        No averaging across epochs is performed.

        The `event_id` column contains the event label obtained from the
        mapping stored in `self.event_id`, rather than the numerical event
        code. Event labels are added to `aperiodic_params` and `gof`, but
        not to `model`.

        See Also
        --------
        AperiodicSpectrumArray.fit_aperiodic_model
        pyrasa.utils.aperiodic_utils.compute_aperiodic_model
        pyrasa.utils.fit_funcs.AbstractFitFun
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
            slope_fit.aperiodic_params['epoch'] = int(self.selection[ix])
            aps_list.append(slope_fit.aperiodic_params.copy())

            slope_fit.gof['epoch'] = int(self.selection[ix])
            slope_fit.gof['event_id'] = event_dict[events[ix]]
            gof_list.append(slope_fit.gof.copy())

            pred_list.append(slope_fit.model)

        return AperiodicFit(aperiodic_params=pd.concat(aps_list), gof=pd.concat(gof_list), model=pd.concat(pred_list))


@define
class IrasaRaw:
    """
    Container for the IRASA decomposition of continuous MNE data.

    Stores the periodic (oscillatory) and aperiodic (fractal) components
    of the power spectrum as MNE-compatible SpectrumArray objects.

    The class provides methods to extract oscillatory peaks, fit parametric
    models to the aperiodic spectrum, and select or remove channels from
    both spectral components.

    Parameters
    ----------
    periodic : PeriodicSpectrumArray
        IRASA-estimated periodic component of the power spectrum.
    aperiodic : AperiodicSpectrumArray
        IRASA-estimated aperiodic component of the power spectrum.

    Notes
    -----
    The periodic and aperiodic spectra must contain the same channels
    in the same order and have identical frequency arrays. This alignment
    is verified during initialization.

    See Also
    --------
    PeriodicSpectrumArray
    AperiodicSpectrumArray
    IrasaEpoched
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
        Extract oscillatory peaks from the IRASA-estimated periodic spectrum.

        Detect oscillatory peaks in the periodic component of the power spectrum
        for each channel. The aperiodic spectrum is used to calculate peak
        power relative to the aperiodic background.

        Peak detection can optionally be performed on a smoothed periodic
        spectrum and restricted to a specified frequency range.

        Parameters
        ----------
        smooth : bool, default True
            Whether to smooth the periodic spectrum before peak detection
            using a Savitzky-Golay filter.
        smoothing_window : int | float, default 2
            Window length of the Savitzky-Golay filter, in Hz.
            Only used when `smooth=True`.
        polyorder : int | None, optional
            Polynomial order of the Savitzky-Golay filter. If None, the
            polynomial order is determined automatically.
        cut_spectrum : tuple[float, float] | None, default (1, 40)
            Lower and upper frequency bounds for peak detection, in Hz.
            If None, the full available frequency range is used.
        peak_threshold : float, default 2.0
            Minimum peak prominence expressed as a multiple of the
            standard deviation of the processed periodic spectrum.
        min_peak_height : float, default 0.0
            Minimum peak height in linear power units.
        min_peak_distance_hz : float | None, optional
            Minimum separation between detected peaks, in Hz.
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

        Raises
        ------
        ValueError
            If the periodic and aperiodic spectra have different channel
            names, frequency arrays, or data shapes.

        Notes
        -----
        Peak detection is performed using `scipy.signal.find_peaks`.
        When smoothing is enabled, relative peak power is calculated
        using the smoothed periodic spectrum and the original
        aperiodic background.

        See Also
        --------
        pyrasa.utils.peak_utils.get_peak_params
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
        Fit a parametric model to the IRASA-estimated aperiodic spectrum.

        Fit the aperiodic component of the power spectrum separately for each
        channel using scipy's nonlinear least-squares optimization.

        Two models are available by default: a fixed model describing a power-law
        relationship without a knee, and a knee model that additionally captures
        a bend in the aperiodic spectrum. Custom fitting functions can be
        provided as subclasses of :class:`~pyrasa.utils.fit_funcs.AbstractFitFun`.

        Parameters
        ----------
        fit_func : str | type[AbstractFitFun], default 'fixed'
            Model used to fit the aperiodic spectrum. Options are:

            - 'fixed': Fit a power-law model without a knee, characterized by
            an offset and exponent.
            - 'knee': Fit a model including a spectral knee, characterized by
            an offset, knee parameter, and two exponents.
            - A subclass of `AbstractFitFun` defining a custom fitting model.

        scale : bool, default False
            Whether to scale the aperiodic spectrum before fitting to improve
            numerical stability. This can be useful when power values are very
            small. The fitted parameters and model predictions are adjusted
            to the original data scale.
        fit_bounds : tuple[float, float] | None, optional
            Lower and upper frequency bounds for model fitting, in Hz.
            If None, the full available frequency range is used.
            Otherwise, only frequencies within the specified bounds are
            included in the fit.

        Returns
        -------
        AperiodicFit
            Object containing three pandas DataFrames:

            - `aperiodic_params`: Fitted model parameters for each channel.
            - `gof`: Goodness-of-fit metrics for each channel, including
            mean squared error, R-squared, AIC, and BIC.
            - `model`: Predicted aperiodic power values for each channel
            and frequency in the fitted range.

        Notes
        -----
        The built-in fitting models operate on logarithmically transformed
        power values and require positive spectral values.

        If the frequency range includes 0 Hz, this frequency is excluded
        from the fitting procedure.

        See Also
        --------
        pyrasa.utils.aperiodic_utils.compute_aperiodic_model
        pyrasa.utils.fit_funcs.AbstractFitFun
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
    Container for the IRASA decomposition of epoched MNE data.

    Stores the periodic (oscillatory) and aperiodic (fractal) components
    of the power spectrum for each epoch and channel as MNE-compatible
    EpochsSpectrumArray objects.

    The class provides methods to extract oscillatory peaks, fit parametric
    models to the aperiodic spectrum, select individual epochs, and select
    or remove channels from both spectral components.

    Parameters
    ----------
    periodic : PeriodicEpochsSpectrum
        IRASA-estimated periodic component of the power spectrum
        for each epoch and channel.
    aperiodic : AperiodicEpochsSpectrum
        IRASA-estimated aperiodic component of the power spectrum
        for each epoch and channel.

    Notes
    -----
    The periodic and aperiodic spectra must have identical channel names,
    frequency arrays, event arrays, epoch selections, and event ID mappings.
    This alignment is verified during initialization.

    See Also
    --------
    PeriodicEpochsSpectrum
    AperiodicEpochsSpectrum
    IrasaRaw
    """

    periodic: PeriodicEpochsSpectrum
    aperiodic: AperiodicEpochsSpectrum

    def __getitem__(
        self,
        item: int | slice | list[int] | list[str] | np.ndarray | str,
    ) -> Self:
        """
        Select epochs from both IRASA spectral components.

        Apply the same epoch selection to the periodic and aperiodic spectra
        using MNE's epoch-selection conventions.

        The operation returns a new IRASA result containing the selected
        epochs without modifying the original object.

        Parameters
        ----------
        item : int | slice | list[int] | list[str] | np.ndarray | str
            Epochs to select. Supports integer indices, slices, lists of
            indices, boolean masks, and event names defined in the
            `event_id` mapping.

        Returns
        -------
        Self
            A new IRASA result containing only the selected epochs.

        See Also
        --------
        mne.time_frequency.EpochsSpectrumArray
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
        Extract oscillatory peaks from the periodic spectrum of each epoch.

        Detect oscillatory peaks separately for each epoch and channel using
        the IRASA-estimated periodic component of the power spectrum.
        The corresponding aperiodic spectrum is used to calculate peak
        power relative to the aperiodic background.

        Peak detection can optionally be performed on a smoothed periodic
        spectrum and restricted to a specified frequency range. The detected
        peaks are combined across epochs and annotated with their corresponding
        experimental event labels.

        Parameters
        ----------
        smooth : bool, default True
            Whether to smooth the periodic spectrum before peak detection
            using a Savitzky-Golay filter.
        smoothing_window : int | float, default 2
            Window length of the Savitzky-Golay filter, in Hz.
            Only used when `smooth=True`.
        polyorder : int | None, optional
            Polynomial order of the Savitzky-Golay filter. If None, the
            polynomial order is determined automatically.
        cut_spectrum : tuple[float, float] | None, default (1, 40)
            Lower and upper frequency bounds for peak detection, in Hz.
            If None, the full available frequency range is used.
        peak_threshold : float, default 2.0
            Minimum peak prominence expressed as a multiple of the
            standard deviation of the processed periodic spectrum.
        min_peak_height : float, default 0.0
            Minimum peak-height threshold passed to `get_peak_params`,
            in linear power units.
        min_peak_distance_hz : float | None, optional
            Minimum separation between detected peaks, in Hz.
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
            - `event_id`: Experimental event label associated with the epoch.
            - `epoch`: Index of the epoch in the original spectrum object,
                       preserved when selecting subsets of epochs.

        Raises
        ------
        ValueError
            If the periodic and aperiodic spectra have different channel
            names, frequency arrays, event arrays, or data shapes.

        Notes
        -----
        Peak detection is performed independently for each epoch and channel
        using `scipy.signal.find_peaks`. No averaging across epochs is performed.

        When smoothing is enabled, relative peak power is calculated using
        the smoothed periodic spectrum and the original aperiodic background.

        The `event_id` column contains the event label obtained from the
        mapping stored in `self.periodic.event_id`, rather than the numerical
        event code.

        See Also
        --------
        IrasaRaw.get_peaks
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
            peak_df['epoch'] = int(self.periodic.selection[ix])
            peak_list.append(peak_df)

        return pd.concat(peak_list, ignore_index=True)

    def fit_aperiodic_model(
        self,
        fit_func: str | type[AbstractFitFun] = 'fixed',
        scale: bool = False,
        fit_bounds: tuple[float, float] | None = None,
    ) -> AperiodicFit:
        """
        Fit a parametric model to the aperiodic spectrum of each epoch.

        Fit the IRASA-estimated aperiodic component of the power spectrum
        separately for each epoch and channel using scipy's nonlinear
        least-squares optimization.

        Two models are available by default: a fixed model describing a
        power-law relationship without a knee, and a knee model that
        additionally captures a bend in the aperiodic spectrum. Custom
        fitting functions can be provided as subclasses of `AbstractFitFun`.

        The fitted parameters and goodness-of-fit metrics are combined across
        epochs and annotated with the corresponding experimental event labels.

        Parameters
        ----------
        fit_func : str | type[AbstractFitFun], default 'fixed'
            Model used to fit the aperiodic spectrum. Options are:

            - 'fixed': Fit a power-law model without a knee, characterized
            by an offset and exponent.
            - 'knee': Fit a model including a spectral knee.
            - A subclass of `AbstractFitFun` defining a custom fitting model.

        scale : bool, default False
            Whether to scale the aperiodic spectrum before fitting to improve
            numerical stability. This can be useful when power values are very
            small. The fitted parameters and model predictions are adjusted
            to the original data scale.
        fit_bounds : tuple[float, float] | None, optional
            Lower and upper frequency bounds for model fitting, in Hz.
            If None, the full available frequency range is used.
            Otherwise, only frequencies within the specified bounds are
            included in the fit.

        Returns
        -------
        AperiodicFit
            Object containing three pandas DataFrames:

            - `aperiodic_params`: Fitted model parameters for each channel
            and epoch, including an `event_id` column with the corresponding
            experimental event label.
            - `gof`: Goodness-of-fit metrics for each channel and epoch,
            including an `event_id` column.
            - `model`: Predicted aperiodic power values for each channel
            and epoch at the fitted frequencies.

        Notes
        -----
        Models are fitted independently for each epoch and channel.
        No averaging across epochs is performed.

        The `event_id` column contains the event label obtained from the
        mapping stored in `self.event_id`, rather than the numerical event
        code. Event labels are added to `aperiodic_params` and `gof`, but
        not to `model`.

        See Also
        --------
        AperiodicSpectrumArray.fit_aperiodic_model
        pyrasa.utils.aperiodic_utils.compute_aperiodic_model
        pyrasa.utils.fit_funcs.AbstractFitFun
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
