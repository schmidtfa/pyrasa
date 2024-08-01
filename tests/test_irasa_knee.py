import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa import irasa
from pyrasa.utils.aperiodic_utils import compute_slope
from pyrasa.utils.peak_utils import get_peak_params

from .settings import EXP_KNEE_COMBO, FS, KNEE_TOLERANCE, MIN_CORR_PSD_CMB, OSC_FREQ, TOLERANCE

# Estimate periodic and aperiodic components with IRASA
# These tests should cover the basic functionality of the IRASA workflow


# knee model
@pytest.mark.parametrize('exponent, knee', EXP_KNEE_COMBO, scope='session')
@pytest.mark.parametrize('fs', FS, scope='session')
def test_irasa_knee_peakless(load_knee_aperiodic_signal, fs, exponent, knee):
    f_range = [0.1, 100]
    irasa_out = irasa(load_knee_aperiodic_signal, fs, f_range, psd_kwargs={'nperseg': 4 * fs})
    # test the shape of the output
    assert irasa_out.freqs.shape[0] == irasa_out.aperiodic.shape[1] == irasa_out.periodic.shape[1]
    freqs_psd, psd = dsp.welch(load_knee_aperiodic_signal, fs, nperseg=int(4 * fs))
    psd_cmb = irasa_out.aperiodic[0, :] + irasa_out.periodic[0, :]
    freq_logical = np.logical_and(freqs_psd >= f_range[0], freqs_psd <= f_range[1])
    r = np.corrcoef(psd[freq_logical], psd_cmb)[0, 1]
    assert r > MIN_CORR_PSD_CMB
    ap_params_k, gof_k = compute_slope(irasa_out.aperiodic, irasa_out.freqs, fit_func='knee')
    ap_params_f, gof_f = compute_slope(irasa_out.aperiodic, irasa_out.freqs, fit_func='fixed')
    # test whether we can get the first exponent correctly
    assert bool(np.isclose(ap_params_k['Exponent_1'][0], 0, atol=TOLERANCE))
    # test whether we can get the second exponent correctly
    assert bool(np.isclose(ap_params_k['Exponent_2'][0], np.abs(exponent), atol=TOLERANCE))
    # test whether we can get the knee correctly
    knee_hat = ap_params_k['Knee'][0] ** (1 / (2 * ap_params_k['Exponent_1'][0] + ap_params_k['Exponent_2'][0]))
    knee_real = knee ** (1 / np.abs(exponent))
    assert bool(np.isclose(knee_hat, knee_real, atol=KNEE_TOLERANCE))
    # test bic/aic -> should be better for knee
    assert gof_k['AIC'][0] < gof_f['AIC'][0]
    assert gof_k['BIC'][0] < gof_f['BIC'][0]


# knee model
@pytest.mark.parametrize('exponent, knee', EXP_KNEE_COMBO, scope='session')
@pytest.mark.parametrize('fs', FS, scope='session')
@pytest.mark.parametrize('osc_freq', OSC_FREQ, scope='session')
def test_irasa_knee_cmb(load_knee_cmb_signal, fs, exponent, knee, osc_freq):
    f_range = [0.1, 100]
    irasa_out = irasa(load_knee_cmb_signal, fs, f_range, psd_kwargs={'nperseg': 4 * fs})
    # test the shape of the output
    assert irasa_out.freqs.shape[0] == irasa_out.aperiodic.shape[1] == irasa_out.periodic.shape[1]
    freqs_psd, psd = dsp.welch(load_knee_cmb_signal, fs, nperseg=int(4 * fs))
    psd_cmb = irasa_out.aperiodic[0, :] + irasa_out.periodic[0, :]
    freq_logical = np.logical_and(freqs_psd >= f_range[0], freqs_psd <= f_range[1])
    r = np.corrcoef(psd[freq_logical], psd_cmb)[0, 1]
    assert r > MIN_CORR_PSD_CMB
    ap_params_k, gof_k = compute_slope(irasa_out.aperiodic, irasa_out.freqs, fit_func='knee')
    ap_params_f, gof_f = compute_slope(irasa_out.aperiodic, irasa_out.freqs, fit_func='fixed')
    # test whether we can get the first exponent correctly
    assert bool(np.isclose(ap_params_k['Exponent_1'][0], 0, atol=TOLERANCE))
    # test whether we can get the second exponent correctly
    assert bool(np.isclose(ap_params_k['Exponent_2'][0], np.abs(exponent), atol=TOLERANCE))
    # test whether we can get the knee correctly
    knee_hat = ap_params_k['Knee'][0] ** (1 / (2 * ap_params_k['Exponent_1'][0] + ap_params_k['Exponent_2'][0]))
    knee_real = knee ** (1 / np.abs(exponent))
    assert bool(np.isclose(knee_hat, knee_real, atol=KNEE_TOLERANCE))
    # test bic/aic -> should be better for knee
    assert gof_k['AIC'][0] < gof_f['AIC'][0]
    assert gof_k['BIC'][0] < gof_f['BIC'][0]
    # test whether we can reconstruct the peak frequency correctly
    pe_params = get_peak_params(irasa_out.periodic, irasa_out.freqs)
    assert bool(np.isclose(np.round(pe_params['cf'], 0), osc_freq))
