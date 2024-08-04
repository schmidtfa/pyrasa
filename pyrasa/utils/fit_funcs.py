import numpy as np
from attrs import define
from scipy.optimize import curve_fit


@define
class AbstractFitFun:
    x: np.ndarray
    y: np.ndarray

    def func(self, *args: float, **kwargs: float) -> np.ndarray:
        pass

    def curve_kwargs(self, *args: float, **kwargs: float) -> dict[str, any]:
        return {}

    def fit_func(self, *args: float, **kwargs: float) -> np.ndarray:
        curve_kwargs = self.curve_kwargs
        p, _ = curve_fit(self.func, self.x, self.y, **curve_kwargs)
        return p


# class AbstractFitFun(abc.ABC):
#     def __init__(self, aperiodic_spectrum: np.ndarray, freq: np.ndarray):
#         self.aperiodic_spectrum = aperiodic_spectrum
#         self.freq = freq

#     @abc.abstractmethod
#     def fit_func(self, *args: float, **kwargs: float) -> np.ndarray:
#         pass

#     @property
#     def curve_kwargs(self) -> dict[str, any]:
#         return {}

#     def __call__(self) -> np.ndarray:
#         p, _ = curve_fit(self.fit_func, self.aperiodic_spectrum, self.freq, **self.curv_kwargs)
#         return p


class FixedFitFun(AbstractFitFun):
    def func(self, x: np.ndarray, b0: float, b: float) -> np.ndarray:
        """
        Specparams fixed fitting function.
        Use this to model aperiodic activity without a spectral knee
        """
        y_hat = b0 - np.log10(x**b)

        return y_hat

    @property
    def curve_kwargs(self) -> dict[str, any]:
        off_guess = self.x[0]
        exp_guess = [np.abs(np.log10(self.x[0] / self.x[-1]) / np.log10(self.y[-1] / self.y[0]))]
        return {
            'maxfev': 10_000,
            'ftol': 1e-5,
            'xtol': 1e-5,
            'gtol': 1e-5,
            'p0': np.array(off_guess + exp_guess),
            'bounds': ((-np.inf, -np.inf), (np.inf, np.inf)),
        }


class KneeFitFun(AbstractFitFun):
    def func(
        self,
        x: np.ndarray,
        b0: float,
        k: float,
        b1: float,
        b2: float,
    ) -> np.ndarray:
        """
        Model aperiodic activity with a spectral knee and a pre-knee slope.
        Use this to model aperiodic activity with a spectral knee
        """
        y_hat = b0 - np.log10(x**b1 * (k + x**b2))

        return y_hat

    @property
    def curve_kwargs(self) -> dict[str, any]:
        off_guess = self.aperiodic_spectrum[0]
        exp_guess = [
            np.abs(
                np.log10(self.aperiodic_spectrum[0] / self.aperiodic_spectrum[-1])
                / np.log10(self.freq[-1] / self.freq[0])
            )
        ]
        cumsum_psd = np.cumsum(self.aperiodic_spectrum)
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
