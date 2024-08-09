import abc
import inspect
from collections.abc import Callable
from typing import Any, ClassVar, no_type_check

import numpy as np
import pandas as pd
from attrs import define
from scipy.optimize import curve_fit


def _get_args(f: Callable) -> list:
    return inspect.getfullargspec(f)[0][2:]


def _get_gof(psd: np.ndarray, psd_pred: np.ndarray, k: int, fit_type: str) -> pd.DataFrame:
    """
    get goodness of fit (i.e. mean squared error and R2)
    BIC and AIC currently assume OLS
    https://machinelearningmastery.com/probabilistic-model-selection-measures/
    """
    # k number of parameters in curve fitting function

    # add np.log10 to psd
    residuals = psd - psd_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((psd - np.mean(psd)) ** 2)

    mse = np.mean(residuals**2)
    n = len(psd)

    bic = n * np.log(mse) + k * np.log(n)
    aic = n * np.log(mse) + 2 * k

    gof = pd.DataFrame({'mse': mse, 'r_squared': 1 - (ss_res / ss_tot), 'BIC': bic, 'AIC': aic}, index=[0])
    gof['fit_type'] = fit_type
    return gof


@define
class AbstractFitFun(abc.ABC):
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

    def fit_func(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        curve_kwargs = self.curve_kwargs
        p, _ = curve_fit(self.func, self.freq, self.aperiodic_spectrum, **curve_kwargs)

        my_args = _get_args(self.func)
        df_params = pd.DataFrame(dict(zip(my_args, p)), index=[0])
        df_params['fit_type'] = self.label

        pred = self.func(self.freq, *p)
        df_gof = _get_gof(self.aperiodic_spectrum, pred, len(p), self.label)

        df_params = self.add_infos_to_df(df_params)
        df_params = self.handle_scaling(df_params, scale_factor=self.scale_factor)

        return df_params, df_gof


class FixedFitFun(AbstractFitFun):
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
