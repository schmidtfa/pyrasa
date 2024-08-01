import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa.utils.aperiodic_utils import compute_slope

from .settings import EXPONENT, FS, MIN_R2, TOLERANCE


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
    slope_fit_f = compute_slope(psd, freqs, fit_func='fixed')
    assert pytest.approx(slope_fit_f.aperiodic_params['Exponent'][0], abs=TOLERANCE) == np.abs(exponent)
    # test goodness of fit should be close to r_squared == 1 for linear model
    assert slope_fit_f.gof['r_squared'][0] > MIN_R2

    # test if we can set fit bounds w/o error
    # _, _ = compute_slope(psd, freqs, fit_func='fixed', fit_bounds=[2, 50])

    # bic and aic for fixed model should be better if linear
    slope_fit_k = compute_slope(psd, freqs, fit_func='knee')
    # assert gof_k['AIC'][0] > gof['AIC'][0]
    assert slope_fit_k.gof['BIC'][0] > slope_fit_f.gof['BIC'][0]

    # test the effect of scaling
    slope_fit_fs = compute_slope(psd, freqs, fit_func='fixed', scale=True)
    assert np.isclose(slope_fit_fs.aperiodic_params['Exponent'], slope_fit_f.aperiodic_params['Exponent'])
    assert np.isclose(slope_fit_fs.gof['r_squared'], slope_fit_f.gof['r_squared'])


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

    match_txt = (
        'The first frequency appears to be 0 this will result in slope fitting problems. '
        + 'Frequencies will be evaluated starting from the next highest in Hz'
    )
    # test bounds too low
    with pytest.raises(AssertionError):
        compute_slope(psd[freq_logical], freqs[freq_logical], fit_func='fixed', fit_bounds=(0, 200))

    # test bounds too high
    with pytest.raises(AssertionError):
        compute_slope(psd[freq_logical], freqs[freq_logical], fit_func='fixed', fit_bounds=(1, 1000))

    # test for warning
    with pytest.warns(UserWarning, match=match_txt):
        compute_slope(psd[freq_logical], freqs[freq_logical], fit_func='fixed')
