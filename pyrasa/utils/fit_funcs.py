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
        - 'lnwBIC': logarithmically sample-size-weighted Bayesian Information Criterion
        - 'lnwAIC': logarithmically sample-size-weighted Akaike Information Criterion
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
    aic = 2 * k + n * np.log(mse)  # + c #real
    aic_log_n = 2 * k + np.log(n) * np.log(
        mse
    )  # works better in simulations than the actual aic in detecting a needed knee model

    # aic = 2 * k + n * mse
    # according to Sclove 1987 only difference between BIC and AIC
    # is that BIC uses log(n) * k instead of 2 * k
    bic = np.log(n) * k + n * np.log(mse)  # + c #real
    bic_log_n = np.log(n) * k + np.log(n) * np.log(
        mse
    )  ## works better in simulations than the actual bic in detecting a needed knee model
    # bic = np.log(n) * k + n * mse
    # Sclove 1987 also hints at sample size adjusted bic
    # an = (n + 2) / 24  # defined in Rissanen 1978 based on minimum-bit representation of a signal
    # abic -> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7299313/
    # abic = np.log(an) * k + np.log(n) * np.log(mse)

    r2 = 1 - (ss_res / ss_tot)
    r2_adj = 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

    gof = pd.DataFrame(
        {'mse': mse, 'R2': r2, 'R2_adj.': r2_adj, 'BIC': bic, 'lnwBIC': bic_log_n, 'AIC': aic, 'lnwAIC': aic_log_n},
        index=[0],
    )
    gof['fit_type'] = fit_type
    return gof


@define
class AbstractFitFun(abc.ABC):
    """
    Abstract base class for fitting parametric models to aperiodic spectra.

    Provides a framework for fitting aperiodic spectral models using
    `scipy.optimize.curve_fit`, including optional logarithmic
    transformations, parameter scaling, and goodness-of-fit evaluation.

    Subclasses must implement `func` to define the mathematical model.
    Additional methods can be overridden to customize fitting parameters,
    include derived model parameters, or handle power scaling.

    Parameters
    ----------
    freq : np.ndarray, shape (n_freqs,)
        Frequency values corresponding to the aperiodic spectrum, in Hz.
    aperiodic_spectrum : np.ndarray, shape (n_freqs,)
        Aperiodic power spectral values to fit.
    scale_factor : int | float
        Factor by which the original power values were multiplied before
        fitting. Used to restore model predictions and, where implemented,
        fitted parameters to the original power scale.

    Attributes
    ----------
    label : ClassVar[str]
        Label identifying the fitting model. Default is 'custom'.
    log10_aperiodic : ClassVar[bool]
        Whether to log10-transform the aperiodic power values before
        fitting. Default is False.
    log10_freq : ClassVar[bool]
        Whether to log10-transform the frequency values before fitting.
        Default is False.

    Notes
    -----
    Subclasses must implement `func` to define the model evaluated by
    `scipy.optimize.curve_fit`.

    The `curve_kwargs` property can be overridden to configure the
    optimization procedure. The `add_infos_to_df` method can be overridden
    to include derived model parameters in the results.

    When `scale_factor` differs from 1, subclasses must implement
    `handle_scaling` to adjust the fitted parameters. Otherwise, a
    `NotImplementedError` is raised.

    Goodness-of-fit metrics are calculated using the spectral values
    in the representation used for fitting, which may be logarithmically
    transformed.
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
        if scale_factor == 1.0:
            return df_params

        raise NotImplementedError(
            'Scaling is not implemented for this fitting function. '
            'Please override handle_scaling() or disable scaling.'
        )

    def fit_func(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit the model and return parameters, goodness-of-fit metrics, and predictions.

        Fit the model defined by `func` using `scipy.optimize.curve_fit`.
        The parameter names are obtained from the signature of `func`.

        Model predictions are transformed back to the original power and
        frequency scales before being returned.

        Returns
        -------
        df_params : pd.DataFrame
            Fitted model parameters, including the `fit_type` column and
            any additional parameters provided by `add_infos_to_df`.
        df_gof : pd.DataFrame
            Goodness-of-fit metrics, including mean squared error (`mse`),
            R-squared (`R2`), adjusted R-squared (`R2_adj.`), AIC, BIC,
            their logarithmically sample-size-weighted variants
            (`lnwAIC`, `lnwBIC`), and `fit_type`.
        df_pred : pd.DataFrame
            Model predictions containing the columns `Frequency (Hz)`,
            `aperiodic_model`, and `fit_type`.

        Notes
        -----
        Goodness-of-fit metrics are computed before transforming model
        predictions back to the original power scale. Consequently, their
        values depend on the representation used for fitting.

        The returned model predictions are adjusted for the specified
        `scale_factor`.
        """
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

        pred /= self.scale_factor

        df_pred = pd.DataFrame({'Frequency (Hz)': freq, 'aperiodic_model': pred})
        df_pred['fit_type'] = self.label

        return df_params, df_gof, df_pred


class FixedFitFun(AbstractFitFun):
    """
    Fit an aperiodic power-law model without a spectral knee.

    This class extends `AbstractFitFun` to model the aperiodic spectrum
    using a fixed power-law relationship.

    The model is defined as:

        log10(P(f)) = Offset - Exponent * log10(f)

    where P(f) is the aperiodic power at frequency f, `Offset` is the
    log10 power at 1 Hz, and `Exponent` describes the power-law decay.

    The model is fitted to log10-transformed power values using
    nonlinear least-squares optimization. Initial parameter estimates
    are obtained from linear regression in log-log space.

    Attributes
    ----------
    label : str
        Model identifier, set to 'fixed'.
    log10_aperiodic : bool
        Whether to log10-transform the aperiodic power values.
        Set to True.

    Notes
    -----
    The model assumes a constant exponent across the fitted frequency
    range and does not account for a spectral knee.

    Optimization uses the Levenberg-Marquardt algorithm with an
    analytical Jacobian.

    See Also
    --------
    AbstractFitFun
    KneeFitFun
    """

    label = 'fixed'
    log10_aperiodic = True

    def func(self, x: np.ndarray, Offset: float, Exponent: float) -> np.ndarray:  # noqa N803
        """
        Evaluate a fixed aperiodic model in log10 power space.

        Parameters
        ----------
        x : np.ndarray
            Frequencies at which to evaluate the model, in Hz.
        Offset : float
            Log10 power at 1 Hz.
        Exponent : float
            Power-law exponent controlling the spectral slope.

        Returns
        -------
        np.ndarray
            Predicted log10 power values at the specified frequencies.
        """
        y_hat = Offset - Exponent * np.log10(x)

        return y_hat

    def handle_scaling(self, df_params: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
        if scale_factor != 1.0:
            df_params['Offset'] -= np.log10(scale_factor)

        return df_params

    def jacobian(self, x: np.ndarray, offset: float, exponent: float) -> np.ndarray:
        """Derivatives with respect to Offset and Exponent."""
        return np.column_stack(
            (
                np.ones_like(x),
                -np.log10(x),
            )
        )

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        slope, intercept = np.polyfit(
            np.log10(self.freq),
            self.aperiodic_spectrum,
            deg=1,
        )  # we could prob also do this instead of the subsequent curve_fit but we want to be consistent

        return {
            **super().curve_kwargs,
            'p0': np.array([intercept, -slope]),
            'method': 'lm',
            'jac': self.jacobian,
            'ftol': 1e-8,
            'xtol': 1e-8,
            'gtol': 1e-8,
        }


class KneeFitFun(AbstractFitFun):
    """
    Fit an aperiodic power-law model with a spectral knee.

    This class extends `AbstractFitFun` to model aperiodic spectra with
    a transition between two power-law regimes.

    The model is defined as:

        log10(P(f)) = Offset
                    - Exponent_1 * log10(f)
                    - log10(Knee + f**Exponent_2)

    where P(f) is the aperiodic power at frequency f.

    `Exponent_1` describes the low-frequency power-law exponent, while
    the high-frequency exponent approaches the sum of `Exponent_1`
    and `Exponent_2`.

    The knee parameter determines the transition between these regimes.
    The corresponding knee frequency is calculated as:

        f_knee = Knee ** (1 / Exponent_2)

    A characteristic timescale is additionally calculated as:

        tau = 1 / (2 * pi * f_knee)

    The model is fitted to log10-transformed power values using bounded
    nonlinear least-squares optimization.

    Attributes
    ----------
    label : str
        Model identifier, set to 'knee'.
    log10_aperiodic : bool
        Whether to log10-transform the aperiodic power values.
        Set to True.

    Notes
    -----
    The knee parameter and both exponents
    are constrained to nonnegative values.

    Initial parameter estimates are selected from multiple candidate
    knee locations and high-frequency exponent contributions.

    The fitted parameters include the derived `Knee Frequency (Hz)`
    and `tau` values. These quantities may be undefined or non-finite
    for degenerate fits, particularly when `Exponent_2` approaches zero.

    See Also
    --------
    AbstractFitFun
    FixedFitFun
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
        Evaluate the knee aperiodic model in log10 power space.

        Parameters
        ----------
        x : np.ndarray
            Frequencies at which to evaluate the model, in Hz.
        Offset : float
            Intercept parameter of the aperiodic model in log10 power space.
        Knee : float
            Nonnegative parameter controlling the spectral knee.
        Exponent_1 : float
            Low-frequency power-law exponent.
        Exponent_2 : float
            Additional exponent determining the change in spectral slope
            above the knee.

        Returns
        -------
        np.ndarray
            Predicted log10 power values at the specified frequencies.
        """
        # y_hat = Offset - np.log10(x**Exponent_1 * (Knee + x**Exponent_2))
        # equivalent yet more stable solution
        y_hat = Offset - Exponent_1 * np.log10(x) - np.log10(Knee + x**Exponent_2)

        return y_hat

    def handle_scaling(self, df_params: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
        if scale_factor != 1.0:
            df_params['Offset'] -= np.log10(scale_factor)

        return df_params

    def add_infos_to_df(self, df_params: pd.DataFrame) -> pd.DataFrame:
        df_params['Knee Frequency (Hz)'] = df_params['Knee'] ** (1.0 / df_params['Exponent_2'])
        df_params['tau'] = 1.0 / (2 * np.pi * df_params['Knee Frequency (Hz)'])
        return df_params

    def jacobian(
        self,
        x: np.ndarray,
        offset: float,
        knee: float,
        exponent_1: float,
        exponent_2: float,
    ) -> np.ndarray:
        """Derivatives in the same order as the model parameters."""
        log_freq = np.log10(x)
        power = x**exponent_2
        denominator = knee + power

        return np.column_stack(
            (
                np.ones_like(x),
                -1.0 / (np.log(10.0) * denominator),
                -log_freq,
                -(power / denominator) * log_freq,
            )
        )

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        log_freq = np.log10(self.freq)
        log_power = self.aperiodic_spectrum

        mean_freq = log_freq.mean()
        centered_freq = log_freq - mean_freq
        spread = centered_freq @ centered_freq

        if not np.isfinite(spread) or spread <= 0:
            raise ValueError('Fitting requires finite positive, distinct frequencies.')

        # Candidate crossover locations, expressed in log10 Hz.
        log_knee_freqs = np.linspace(log_freq.min(), log_freq.max(), 13)

        # Starting candidates only, not bounds on the final fit.
        exponent_candidates = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

        best_error = np.inf
        p0 = None

        min_log = np.log10(np.finfo(float).tiny)
        max_log = np.log10(np.finfo(float).max)

        for exponent_2 in exponent_candidates:
            for log_knee_freq in log_knee_freqs:
                # Knee = crossover_frequency ** Exponent_2
                log_knee = exponent_2 * log_knee_freq

                if not min_log < log_knee < max_log:
                    continue

                # log10(Knee + f**Exponent_2), evaluated stably.
                log_denominator = np.logaddexp(
                    np.log(10.0) * log_knee,
                    np.log(10.0) * exponent_2 * log_freq,
                ) / np.log(10.0)

                adjusted_power = log_power + log_denominator
                mean_power = adjusted_power.mean()

                # Conditional least-squares estimate, with Exponent_1 >= 0.
                exponent_1 = max(
                    0.0,
                    -np.dot(
                        centered_freq,
                        adjusted_power - mean_power,
                    )
                    / spread,
                )

                offset = mean_power + exponent_1 * mean_freq

                residual = mean_power - exponent_1 * centered_freq - adjusted_power
                error = residual @ residual

                if error < best_error:
                    best_error = error
                    p0 = np.array(
                        [
                            offset,
                            10.0**log_knee,
                            exponent_1,
                            exponent_2,
                        ]
                    )

        if p0 is None:
            raise ValueError('No finite initial guess found.')

        return {
            **super().curve_kwargs,
            'p0': p0,
            'bounds': (
                (-np.inf, 0.0, 0.0, 0.0),
                (np.inf, np.inf, np.inf, np.inf),
            ),
            'method': 'trf',
            'jac': self.jacobian,
            'x_scale': 'jac',
            'ftol': 1e-8,
            'xtol': 1e-8,
            'gtol': 1e-8,
        }
