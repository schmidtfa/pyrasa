import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa import irasa
from pyrasa.utils.aperiodic_utils import compute_aperiodic_model
from pyrasa.utils.fit_funcs import AbstractFitFun

from .settings import EXPONENT, FS, HIGH_TOLERANCE, MIN_R2, TOLERANCE


# Test slope fitting functionality
@pytest.mark.parametrize('exponent', EXPONENT, scope='session')
@pytest.mark.parametrize('fs', FS, scope='session')
# @pytest.mark.xfail
def test_slope_fitting_fixed(fixed_aperiodic_signal, fs, exponent):
    f_range = [1, 100]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(fixed_aperiodic_signal, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    freqs, psd = freqs[freq_logical], psd[freq_logical]

    # test whether we can reconstruct the exponent correctly
    aperiodic_fit_f = compute_aperiodic_model(psd, freqs, fit_func='fixed')
    assert pytest.approx(aperiodic_fit_f.aperiodic_params['Exponent'][0], abs=TOLERANCE) == np.abs(exponent)
    # test goodness of fit should be close to r_squared == 1 for linear model
    assert aperiodic_fit_f.gof['R2'][0] > MIN_R2

    # test if we can set fit bounds w/o error
    # _, _ = compute_slope(psd, freqs, fit_func='fixed', fit_bounds=[2, 50])

    # bic and aic for fixed model should be better if linear
    aperiodic_fit_k = compute_aperiodic_model(psd, freqs, fit_func='knee')
    # assert gof_k['AIC'][0] > gof['AIC'][0]
    assert aperiodic_fit_k.gof['BIC'][0] > aperiodic_fit_f.gof['BIC'][0]

    # test the effect of scaling
    aperiodic_fit_fs = compute_aperiodic_model(psd, freqs, fit_func='fixed', scale=True)
    assert np.isclose(aperiodic_fit_fs.aperiodic_params['Exponent'], aperiodic_fit_f.aperiodic_params['Exponent'])
    assert np.isclose(aperiodic_fit_fs.gof['R2'], aperiodic_fit_f.gof['R2'])


@pytest.mark.parametrize('exponent, fs', [(-1, 500)], scope='session')
def test_slope_fitting_settings(
    fixed_aperiodic_signal,
    exponent,
    fs,
):
    f_range = [0, 100]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(fixed_aperiodic_signal, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])

    # match_txt = (
    #     'The first frequency appears to be 0 this will result in slope fitting problems. '
    #     + 'Frequencies will be evaluated starting from the next highest in Hz'
    # )
    # test bounds too low
    with pytest.raises(AssertionError):
        compute_aperiodic_model(psd[freq_logical], freqs[freq_logical], fit_func='fixed', fit_bounds=(0, 200))

    # test bounds too high
    with pytest.raises(AssertionError):
        compute_aperiodic_model(psd[freq_logical], freqs[freq_logical], fit_func='fixed', fit_bounds=(1, 1000))

    # test bounds correct
    compute_aperiodic_model(psd[freq_logical], freqs[freq_logical], fit_func='fixed', fit_bounds=(5, 40))

    # test for warning
    with pytest.warns(UserWarning):  # , match=match_txt):
        compute_aperiodic_model(psd[freq_logical], freqs[freq_logical], fit_func='fixed')

    # test misspecify string in fit_func
    with pytest.raises(ValueError):
        compute_aperiodic_model(psd[freq_logical], freqs[freq_logical], fit_func='incredible', fit_bounds=(5, 40))


# test custom slope fitting functions
@pytest.mark.parametrize('exponent, fs', [(-1, 500)], scope='session')
def test_custom_slope_fitting(
    fixed_aperiodic_signal,
    exponent,
    fs,
):
    f_range = [1.5, 100]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(fixed_aperiodic_signal, fs, nperseg=int(4 * fs))
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
        def curve_kwargs(self) -> dict[str, any]:
            aperiodic_nolog = self.aperiodic_spectrum
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

    aperiodic_fit = compute_aperiodic_model(np.log10(psd), np.log10(freqs), fit_func=CustomFitFun)
    # add a high tolerance
    assert pytest.approx(np.abs(aperiodic_fit.aperiodic_params['b'][0]), abs=HIGH_TOLERANCE) == np.abs(exponent)

    class CustomFitFun(AbstractFitFun):
        log10_aperiodic = True
        log10_freq = True

        def func(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
            """
            Specparams fixed fitting function.
            Use this to model aperiodic activity without a spectral knee
            """
            y_hat = a + b * x

            return y_hat

    irasa_spectrum = irasa(fixed_aperiodic_signal, fs, f_range, psd_kwargs={'nperseg': 4 * fs})
    aperiodic_fit = irasa_spectrum.fit_aperiodic_model(fit_func=CustomFitFun)

    # add a high tolerance
    assert pytest.approx(np.abs(aperiodic_fit.aperiodic_params['b'][0]), abs=HIGH_TOLERANCE) == np.abs(exponent)
