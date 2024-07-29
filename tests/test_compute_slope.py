import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa.utils.aperiodic_utils import compute_slope

from .settings import EXPONENT, FS, MIN_R2, TOLERANCE


# Test slope fitting functionality
@pytest.mark.parametrize('exponent', EXPONENT, scope='session')
@pytest.mark.parametrize('fs', FS, scope='session')
def test_slope_fitting_fixed(fixed_aperiodic_signal, fs, exponent):
    f_range = [1, 250]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(fixed_aperiodic_signal, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    freqs, psd = freqs[freq_logical], psd[freq_logical]
    # test whether we can reconstruct the exponent correctly
    ap_params, gof = compute_slope(psd, freqs, fit_func='fixed')
    assert bool(np.isclose(ap_params['Exponent'][0], np.abs(exponent), atol=TOLERANCE))
    # test goodness of fit should be close to r_squared == 1 for linear model
    assert gof['r_squared'][0] > MIN_R2


# Takes too long need to pregenerate
# @pytest.mark.parametrize('exponent', EXPONENT, scope='session')
# @pytest.mark.parametrize('fs', FS, scope='session')
# @pytest.mark.parametrize('knee_freq', KNEE_FREQ, scope='session')
# def test_slope_fitting_knee(knee_aperiodic_signal, fs, exponent, knee_freq):

#     f_range = [1, 250]
#     #test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
#     freqs, psd = dsp.welch(knee_aperiodic_signal, fs, nperseg=int(4*fs))
#     freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
#     freqs, psd = freqs[freq_logical],  psd[freq_logical]
#     #test whether we can reconstruct the exponent correctly
#     ap_params, gof = compute_slope(psd, freqs, fit_func='knee')
#     assert bool(np.isclose(ap_params['Exponent_1'][0], 0, atol=TOLERANCE))
#     assert bool(np.isclose(ap_params['Exponent_2'][0], np.abs(exponent), atol=TOLERANCE))
#     assert bool(np.isclose(ap_params['Knee Frequency (Hz)'][0], knee_freq, atol=TOLERANCE))
#     #test goodness of fit
#     assert gof['r_squared'][0] > MIN_R2
