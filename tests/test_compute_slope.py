import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa.utils.aperiodic_utils import compute_slope

from .settings import EXPONENT, FS, KNEE_FREQ, KNEE_TOLERANCE, MIN_R2, TOLERANCE


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
    ap_params_f, gof_f = compute_slope(psd, freqs, fit_func='fixed')
    assert pytest.approx(ap_params_f['Exponent'][0], abs=TOLERANCE) == np.abs(exponent)
    # test goodness of fit should be close to r_squared == 1 for linear model
    assert gof_f['r_squared'][0] > MIN_R2

    # test if we can set fit bounds w/o error
    # _, _ = compute_slope(psd, freqs, fit_func='fixed', fit_bounds=[2, 50])

    # bic and aic for fixed model should be better if linear
    ap_params_k, gof_k = compute_slope(psd, freqs, fit_func='knee')
    # assert gof_k['AIC'][0] > gof['AIC'][0]
    assert gof_k['BIC'][0] > gof_f['BIC'][0]

    # test the effect of scaling
    ap_params_fs, gof_fs = compute_slope(psd, freqs, fit_func='fixed', scale=True)
    assert np.isclose(ap_params_fs['Exponent'], ap_params_f['Exponent'])
    assert np.isclose(gof_fs['r_squared'], gof_f['r_squared'])


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


# Takes too long need to pregenerate
@pytest.mark.parametrize('exponent, fs, knee_freq', [(-1, 500, 15)], scope='session')
def test_slope_fitting_knee(knee_aperiodic_signal, fs, exponent):
    f_range = [1, 200]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(knee_aperiodic_signal, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    freqs, psd = freqs[freq_logical], psd[freq_logical]
    # test whether we can reconstruct the exponent correctly
    ap_params_k, gof_k = compute_slope(psd, freqs, fit_func='knee')
    ap_params_f, gof_f = compute_slope(psd, freqs, fit_func='fixed')
    # assert pytest.approx(ap_params_k['Exponent_1'][0], abs=TOLERANCE) == 0
    assert bool(np.isclose(ap_params_k['Exponent_2'][0], np.abs(exponent), atol=TOLERANCE))
    assert bool(np.isclose(ap_params_k['Knee Frequency (Hz)'][0], KNEE_FREQ, atol=KNEE_TOLERANCE))
    # test goodness of fit
    assert gof_k['r_squared'][0] > MIN_R2
    assert gof_k['r_squared'][0] > gof_f['r_squared'][0]  # r2 for knee model should be higher than knee if knee
    # bic and aic for knee model should be better if knee
    assert gof_k['AIC'][0] < gof_f['AIC'][0]
    assert gof_k['BIC'][0] < gof_f['BIC'][0]


# #pytest.mark.usefixtures(['exponent', 'fs'], [EXPONENT[0], FS[0]], scope='session')
# @pytest.mark.parametrize('exponent', EXPONENT, scope='session')
# @pytest.mark.parametrize('fs', FS, scope='session')
# def test_slope_fitting_settings(fixed_aperiodic_signal, fs, exponent):
#     f_range = [0, 100]
#     # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
#     freqs, psd = dsp.welch(fixed_aperiodic_signal, 500, nperseg=int(4 * 500))
#     freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
#     ap_params, gof = compute_slope(psd, freqs, fit_func='fixed')
#     ap_params, gof = compute_slope(psd, freqs, fit_func='fixed', fit_bound=(0, 40))
#     ap_params, gof = compute_slope(psd, freqs, fit_func='fixed', fit_bound=(-1, 500))
