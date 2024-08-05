#%%
import scipy.signal as dsp
from pyrasa.utils.aperiodic_utils import compute_slope
from pyrasa.utils.fit_funcs import AbstractFitFun
import numpy as np
from neurodsp.sim import sim_powerlaw
from typing import Any

n_secs = 60
fs=500
f_range = [1.5, 300]
exponent = -1.5

sig = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exponent)

# test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
freqs, psd = dsp.welch(sig, fs, nperseg=int(4 * fs))
freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
psd, freqs = psd[freq_logical], freqs[freq_logical]


class CustomFitFun(AbstractFitFun):
    def func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Specparams fixed fitting function.
        Use this to model aperiodic activity without a spectral knee
        """
        y_hat = a + b * x

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

#%%
slope_fit = compute_slope(np.log10(psd), np.log10(freqs), fit_func=CustomFitFun)
# %%
