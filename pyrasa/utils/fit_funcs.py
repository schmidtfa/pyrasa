"""Classes used to model aperiodic spectra"""

import abc
import inspect
from collections.abc import Callable
from typing import Any, ClassVar, no_type_check

import numpy as np
import pandas as pd
from attrs import define
from scipy.optimize import curve_fit


def _get_args(f: Callable) -> list:
    """
    Extracts the argument names from a function, excluding the first two.

    Parameters
    ----------
    f : Callable
        The function or method from which to extract argument names.

    Returns
    -------
    list
        A list of argument names, excluding the first two.
    """

    return inspect.getfullargspec(f)[0][2:]


def _get_gof(psd: np.ndarray, psd_pred: np.ndarray, k: int, fit_type: str) -> pd.DataFrame:
    """
    Calculate the goodness of fit metrics for a given model prediction against
    actual aperiodic power spectral density (PSD) data.

    This function computes several statistics to evaluate how well the predicted PSD values
    match the observed PSD values. The metrics include Mean Squared Error (MSE), R-squared (R²),
    Bayesian Information Criterion (BIC), and Akaike Information Criterion (AIC).

    Parameters
    ----------
    psd : np.ndarray
        The observed power spectral density values.
    psd_pred : np.ndarray
        The predicted power spectral density values from the model.
    k : int
        The number of parameters in the curve fitting function used to predict the `psd`.
    fit_type : str
        A description or label for the type of fit/model used, which will be included in the output DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the goodness of fit metrics:
        - 'mse': Mean Squared Error
        - 'r_squared': R-squared value
        - 'BIC': Bayesian Information Criterion
        - 'AIC': Akaike Information Criterion
        - 'fit_type': The type of fit/model used (provided as input)

    Notes
    -----
    - BIC and AIC calculations currently assume Ordinary Least Squares (OLS) regression.

    References
    ----------
    For further details on BIC and AIC, see: https://machinelearningmastery.com/probabilistic-model-selection-measures/
    """

    residuals = psd - psd_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((psd - np.mean(psd)) ** 2)

    mse = np.mean(residuals**2)
    n = len(psd)

    # https://robjhyndman.com/hyndsight/lm_aic.html
    # c is in practice sometimes dropped. Only relevant when comparing models with different n
    # c = n + n * np.log(2 * np.pi)
    # aic = 2 * k + n * np.log(mse) + c #real
    aic = 2 * k + np.log(n) * np.log(mse)  # + c
    # aic = 2 * k + n * mse
    # according to Sclove 1987 only difference between BIC and AIC
    # is that BIC uses log(n) * k instead of 2 * k
    # bic = np.log(n) * k + n * np.log(mse) + c #real
    bic = np.log(n) * k + np.log(n) * np.log(mse)  # + c
    # bic = np.log(n) * k + n * mse
    # Sclove 1987 also hints at sample size adjusted bic
    an = (n + 2) / 24  # defined in Rissanen 1978 based on minimum-bit representation of a signal
    # abic -> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7299313/
    abic = np.log(an) * k + np.log(n) * np.log(mse)

    r2 = 1 - (ss_res / ss_tot)
    r2_adj = 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

    gof = pd.DataFrame({'mse': mse, 'R2': r2, 'R2_adj.': r2_adj, 'BIC': bic, 'BIC_adj.': abic, 'AIC': aic}, index=[0])
    gof['fit_type'] = fit_type
    return gof


@define
class AbstractFitFun(abc.ABC):
    """
    Abstract base class for fitting functions used to model aperiodic spectra.

    This class provides a framework for defining and fitting models to aperiodic spectra.
    It handles common functionality required for fitting a model, such as scaling and goodness-of-fit
    computation. Subclasses should implement the `func` method to define the specific fitting function
    used for curve fitting.

    Attributes
    ----------
    freq : np.ndarray
        The frequency values associated with the aperiodic spectrum data.
    aperiodic_spectrum : np.ndarray
        The aperiodic spectrum data to which the model will be fit.
    scale_factor : int | float
        A scaling factor used to adjust the fit results.
    label : ClassVar[str]
        A label to identify the type of fit or model used. Default is 'custom'.
    log10_aperiodic : ClassVar[bool]
        If True, the aperiodic spectrum values will be transformed using log10. Default is False.
    log10_freq : ClassVar[bool]
        If True, the frequency values will be transformed using log10. Default is False.

    Methods
    -------
    __attrs_post_init__()
        Post-initialization method to apply log10 transformations if specified.
    func(x: np.ndarray, *args: float) -> np.ndarray
        Abstract method to define the model function. Must be implemented by subclasses
        and should be applicable to scipy.optimize.curve_fit.
    curve_kwargs() -> dict[str, Any]
        Returns keyword arguments for the curve fitting process.
    add_infos_to_df(df_params: pd.DataFrame) -> pd.DataFrame
        Method to add additional information to the parameters DataFrame. Can be overridden by subclasses.
    handle_scaling(df_params: pd.DataFrame, scale_factor: float) -> pd.DataFrame
        Adjusts the parameters DataFrame based on the scaling factor. Can be overridden by subclasses.
    fit_func() -> tuple[pd.DataFrame, pd.DataFrame]
        Fits the model to the data and returns DataFrames containing the model parameters and goodness-of-fit metrics.

    Notes
    -----
    - Subclasses must implement the `func` method to define the model's functional form.
    - The `curve_kwargs` method can be overridden to customize curve fitting options.
    - The `add_infos_to_df` and `handle_scaling` methods are intended to be overridden if additional
      functionality or specific scaling behavior is required.

    References
    ----------
    For details on goodness-of-fit metrics and their calculations, see the documentation for `_get_gof`.
    """

    freq: np.ndarray
    aperiodic_spectrum: np.ndarray
    scale_factor: int | float
    label: ClassVar[str] = 'custom'
    log10_aperiodic: ClassVar[bool] = False
    log10_freq: ClassVar[bool] = False

    def __attrs_post_init__(self) -> None:
        if self.log10_aperiodic:
            self.aperiodic_spectrum = np.log10(self.aperiodic_spectrum)
        if self.log10_freq:
            self.freq = np.log10(self.freq)

    @abc.abstractmethod
    @no_type_check
    def func(self, x: np.ndarray, *args: float) -> np.ndarray:
        pass

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        return {
            'maxfev': 10_000,
            'ftol': 1e-5,
            'xtol': 1e-5,
            'gtol': 1e-5,
        }

    def add_infos_to_df(self, df_params: pd.DataFrame) -> pd.DataFrame:
        return df_params

    def handle_scaling(self, df_params: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
        if 'Offset' in df_params.columns:
            df_params['Offset'] /= scale_factor
        elif scale_factor != 1.0:
            raise ValueError('Scale Factor not handled. You need to overwrite the handle_scaling method.')
        return df_params

    def fit_func(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        curve_kwargs = self.curve_kwargs
        p, _ = curve_fit(self.func, self.freq, self.aperiodic_spectrum, **curve_kwargs)

        my_args = _get_args(self.func)
        df_params = pd.DataFrame(dict(zip(my_args, p)), index=[0])
        df_params['fit_type'] = self.label

        pred = self.func(self.freq, *p)
        df_gof = _get_gof(self.aperiodic_spectrum, pred, len(p), self.label)

        df_params = self.add_infos_to_df(df_params)
        df_params = self.handle_scaling(df_params, scale_factor=self.scale_factor)

        freq = self.freq.copy()
        if self.log10_aperiodic:
            pred = 10**pred
        if self.log10_freq:
            freq = 10**freq
        df_pred = pd.DataFrame({'Frequency (Hz)': freq, 'aperiodic_model': pred})
        df_pred['fit_type'] = self.label

        return df_params, df_gof, df_pred


class FixedFitFun(AbstractFitFun):
    """
    A model for fitting aperiodic activity in power spectra.

    The `FixedFitFun` class extends `AbstractFitFun` to model aperiodic activity in power spectra
    using a fixed function that does not include a spectral knee. This model is suitable for
    cases where the aperiodic component of the spectrum follows a consistent slope across
    the entire frequency range.

    Attributes
    ----------
    label : str
        A label to identify this fitting model. Default is 'fixed'.
    log10_aperiodic : bool
        Indicates whether to log-transform the aperiodic spectrum. Default is True.

    Methods
    -------
    func(x: np.ndarray, Offset: float, Exponent: float) -> np.ndarray
        Defines the model function for aperiodic activity without a spectral knee.

    curve_kwargs() -> dict[str, Any]
        Generates initial guess parameters and other keyword arguments for curve fitting.
    """

    label = 'fixed'
    log10_aperiodic = True

    def func(self, x: np.ndarray, Offset: float, Exponent: float) -> np.ndarray:  # noqa N803
        """
        Specparams fixed fitting function.
        Use this to model aperiodic activity without a spectral knee
        """
        y_hat = Offset - np.log10(x**Exponent)

        return y_hat

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        aperiodic_nolog = 10**self.aperiodic_spectrum
        off_guess = [aperiodic_nolog[0]]
        exp_guess = [
            np.abs(np.log10(aperiodic_nolog[0] / aperiodic_nolog[-1]) / np.log10(self.freq[-1] / self.freq[0]))
        ]
        return {
            'maxfev': 10_000,
            'ftol': 1e-5,
            'xtol': 1e-5,
            'gtol': 1e-5,
            'p0': np.array(off_guess + exp_guess),
            'bounds': ((-np.inf, -np.inf), (np.inf, np.inf)),
        }


class KneeFitFun(AbstractFitFun):
    """
    A model for fitting aperiodic activity in power spectra with a spectral knee.

    The `KneeFitFun` class extends `AbstractFitFun` to model aperiodic activity in power spectra
    using a function that includes a spectral knee. This model is particularly useful for
    cases where the aperiodic component of the spectrum has a break or knee, representing
    a transition between two different spectral slopes.

    Attributes
    ----------
    label : str
        A label to identify this fitting model. Default is 'knee'.
    log10_aperiodic : bool
        Indicates whether to log-transform the aperiodic spectrum. Default is True.

    Methods
    -------
    func(x: np.ndarray, Offset: float, Knee: float, Exponent_1: float, Exponent_2: float) -> np.ndarray
        Defines the model function for aperiodic activity with a spectral knee and pre-knee slope.

    add_infos_to_df(df_params: pd.DataFrame) -> pd.DataFrame
        Adds calculated knee frequency to the DataFrame of fit parameters.

    curve_kwargs() -> dict[str, Any]
        Generates initial guess parameters and other keyword arguments for curve fitting.
    """

    label = 'knee'
    log10_aperiodic = True

    def func(
        self,
        x: np.ndarray,
        Offset: float,  # noqa N803
        Knee: float,  # noqa N803
        Exponent_1: float,  # noqa N803
        Exponent_2: float,  # noqa N803
    ) -> np.ndarray:
        """
        Model aperiodic activity with a spectral knee and a pre-knee slope.
        Use this to model aperiodic activity with a spectral knee
        """
        y_hat = Offset - np.log10(x**Exponent_1 * (Knee + x**Exponent_2))

        return y_hat

    def add_infos_to_df(self, df_params: pd.DataFrame) -> pd.DataFrame:
        df_params['Knee Frequency (Hz)'] = df_params['Knee'] ** (
            1.0 / (2 * df_params['Exponent_1'] + df_params['Exponent_2'])
        )
        df_params['tau'] = 1.0 / (2 * np.pi * df_params['Knee Frequency (Hz)'])
        return df_params

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        aperiodic_nolog = 10**self.aperiodic_spectrum
        off_guess = [aperiodic_nolog[0]]
        exp_guess = [
            np.abs(np.log10(aperiodic_nolog[0] / aperiodic_nolog[-1]) / np.log10(self.freq[-1] / self.freq[0]))
        ]
        cumsum_psd = np.cumsum(aperiodic_nolog)
        half_pw_freq = self.freq[np.abs(cumsum_psd - (0.5 * cumsum_psd[-1])).argmin()]
        # make the knee guess the point where we have half the power in the spectrum seems plausible to me
        knee_guess = [half_pw_freq ** (exp_guess[0] + exp_guess[0])]

        return {
            'maxfev': 10_000,
            'ftol': 1e-5,
            'xtol': 1e-5,
            'gtol': 1e-5,
            'p0': np.array(off_guess + knee_guess + exp_guess + exp_guess),
            'bounds': ((0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf)),
        }
