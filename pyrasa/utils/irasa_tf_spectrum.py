import numpy as np
import pandas as pd
from attrs import define

from pyrasa.utils.aperiodic_utils import compute_slope_sprint
from pyrasa.utils.peak_utils import get_peak_params_sprint
from pyrasa.utils.types import SlopeFit

min_ndim = 2


@define
class IrasaTfSpectrum:
    freqs: np.ndarray
    time: np.ndarray
    raw_spectrum: np.ndarray
    aperiodic: np.ndarray
    periodic: np.ndarray
    ch_names: np.ndarray | None

    def get_slopes(
        self, fit_func: str = 'fixed', scale: bool = False, fit_bounds: tuple[float, float] | None = None
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

        Returns:    SlopeFit
                        df_aps: DataFrame
                            DataFrame containing the center frequency, bandwidth and peak height for each channel
                        df_gof: DataFrame
                            DataFrame containing the goodness of fit of the specific fit function for each channel.

        """
        return compute_slope_sprint(
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
