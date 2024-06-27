#%% inherit from spectrum array
from mne.time_frequency import SpectrumArray

class PeriodicSpectrumArray(SpectrumArray):

    ''' Childclass of SpectrumArray '''

    def __init__(
        self,
        data,
        info,
        freqs,
        *,
        verbose=None,
    ):
        #_check_data_shape(data, freqs, info, ndim=2)

        self.__setstate__(
            dict(
                method="unknown",
                data=data,
                sfreq=info["sfreq"],
                dims=("channel", "freq"),
                freqs=freqs,
                inst_type_str="Raw",
                data_type="Periodic Power Spectrum",
                info=info,
            )
        )

    def plot(self, *, picks=None, average=False, dB=False,
        amplitude=False, xscale="linear", ci="sd",
        ci_alpha=0.3, color="black", alpha=None,
        spatial_colors=True, sphere=None, exclude=(),
        axes=None, show=True,):
        
        super().plot(picks=picks, average=average, dB=dB,
                     amplitude=amplitude, xscale=xscale, ci=ci,
                     ci_alpha=ci_alpha, color=color, alpha=alpha,
                     spatial_colors=spatial_colors, sphere=sphere, exclude=exclude,
                     axes=axes, show=show,)
        
    def plot_topo(self, *, dB=False, layout=None,
        color="w", fig_facecolor="k", axis_facecolor="k",
        axes=None, block=False, show=True,):

        super().plot_topo(dB=dB, layout=layout, color=color, 
                          fig_facecolor=fig_facecolor, axis_facecolor=axis_facecolor,
                          axes=axes, block=block, show=show,)
        
    def get_peaks(self,
                  smoothing_window=1,
                  cut_spectrum=(1, 40),
                  peak_threshold=2.5,
                  min_peak_height=0.0,
                  peak_width_limits=(.5, 12)):

        '''
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

        '''
        
        from peak_utils import get_peak_params
        peak_df = get_peak_params(self.get_data(),
                                  self.freqs,
                                  self.ch_names,
                                  smoothing_window=smoothing_window,
                                  cut_spectrum=cut_spectrum,
                                  peak_threshold=peak_threshold,
                                  min_peak_height=min_peak_height,
                                  peak_width_limits=peak_width_limits)
        
        return peak_df
    

class AperiodicSpectrumArray(SpectrumArray):

    def __init__(
        self,
        data,
        info,
        freqs,
        *,
        verbose=None,
    ):
        #_check_data_shape(data, freqs, info, ndim=2)

        self.__setstate__(
            dict(
                method="unknown",
                data=data,
                sfreq=info["sfreq"],
                dims=("channel", "freq"),
                freqs=freqs,
                inst_type_str="Raw",
                data_type="Aperiodic Power Spectrum",
                info=info,
            )
        )

    def get_slope(self, fit_func='fixed', fit_bounds=None):

        '''
        This method can be used to extract aperiodic parameters from the aperiodic spectrum extracted from IRASA.
        The algorithm works by applying one of two different curve fit functions and returns the associated parameters,
        as well as the respective goodness of fit.

        Parameters: 
                    fit_func : string
                        Can be either "fixed" or "knee".
                    fit_bounds : None, tuple
                        Lower and upper bound for the fit function, should be None if the whole frequency range is desired.
                        Otherwise a tuple of (lower, upper)

        Returns:    df_aps: DataFrame
                        DataFrame containing the center frequency, bandwidth and peak height for each channel
                    df_gof: DataFrame
                        DataFrame containing the goodness of fit of the specific fit function for each channel.

        '''

        from aperiodic_utils import compute_slope
        df_aps, df_gof = compute_slope(self.get_data(),
                                       self.freqs,
                                       ch_names=self.ch_names,
                                       fit_func=fit_func,
                                       fit_bounds=fit_bounds,
                                       )
        
        return df_aps, df_gof