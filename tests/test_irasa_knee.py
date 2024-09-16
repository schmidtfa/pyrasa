import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa import irasa, irasa_sprint

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
    slope_fit_k = irasa_out.fit_aperiodic_model(fit_func='knee')
    slope_fit_f = irasa_out.fit_aperiodic_model(fit_func='fixed')
    # test whether we can get the first exponent correctly
    assert bool(np.isclose(slope_fit_k.aperiodic_params['Exponent_1'][0], 0, atol=TOLERANCE))
    # test whether we can get the second exponent correctly
    assert bool(
        np.isclose(
            slope_fit_k.aperiodic_params['Exponent_2'][0],
            np.abs(exponent),
            atol=TOLERANCE,
        )
    )
    # test whether we can get the knee correctly
    knee_hat = slope_fit_k.aperiodic_params['Knee'][0] ** (
        1 / (2 * slope_fit_k.aperiodic_params['Exponent_1'][0] + slope_fit_k.aperiodic_params['Exponent_2'][0])
    )
    knee_real = knee ** (1 / np.abs(exponent))
    assert bool(np.isclose(knee_hat, knee_real, atol=KNEE_TOLERANCE))
    # test bic/aic -> should be better for knee
    assert slope_fit_k.gof['AIC'][0] < slope_fit_f.gof['AIC'][0]
    assert slope_fit_k.gof['BIC_adj.'][0] < slope_fit_f.gof['BIC_adj.'][0]


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
    slope_fit_k = irasa_out.fit_aperiodic_model(fit_func='knee')
    slope_fit_f = irasa_out.fit_aperiodic_model(fit_func='fixed')
    # test whether we can get the first exponent correctly
    assert bool(np.isclose(slope_fit_k.aperiodic_params['Exponent_1'][0], 0, atol=TOLERANCE))
    # test whether we can get the second exponent correctly
    assert bool(
        np.isclose(
            slope_fit_k.aperiodic_params['Exponent_2'][0],
            np.abs(exponent),
            atol=TOLERANCE,
        )
    )
    # test whether we can get the knee correctly
    knee_hat = slope_fit_k.aperiodic_params['Knee'][0] ** (
        1 / (2 * slope_fit_k.aperiodic_params['Exponent_1'][0] + slope_fit_k.aperiodic_params['Exponent_2'][0])
    )
    knee_real = knee ** (1 / np.abs(exponent))
    assert bool(np.isclose(knee_hat, knee_real, atol=KNEE_TOLERANCE))
    # test bic/aic -> should be better for knee
    assert slope_fit_k.gof['AIC'][0] < slope_fit_f.gof['AIC'][0]
    assert slope_fit_k.gof['BIC_adj.'][0] < slope_fit_f.gof['BIC_adj.'][0]
    # test whether we can reconstruct the peak frequency correctly
    pe_params = irasa_out.get_peaks()
    assert bool(np.isclose(np.round(pe_params['cf'], 0), osc_freq))


@pytest.mark.parametrize('exponent, knee', [(-1.5, 1000)], scope='session')
@pytest.mark.parametrize('fs', [1000], scope='session')
@pytest.mark.parametrize('osc_freq', [10], scope='session')
def test_aperiodic_error(load_knee_cmb_signal, fs, exponent, knee, osc_freq):
    duration = 4
    overlap = 0.5
    irasa_out = irasa(
        load_knee_cmb_signal,
        fs=fs,
        band=(0.1, 50),
        psd_kwargs={'nperseg': duration * fs, 'noverlap': duration * fs * overlap},
        hset_info=(1, 2.0, 0.05),
    )

    irasa_out_bad = irasa(
        load_knee_cmb_signal,
        fs=fs,
        band=(0.1, 50),
        psd_kwargs={'nperseg': duration * fs, 'noverlap': duration * fs * overlap},
        hset_info=(1, 8.0, 0.05),
    )

    assert np.mean(irasa_out.get_aperiodic_error()) < np.mean(irasa_out_bad.get_aperiodic_error())


@pytest.mark.parametrize('fs', [1000], scope='session')
@pytest.mark.parametrize('exponent_1', [-0], scope='session')
@pytest.mark.parametrize('exponent_2', [-2], scope='session')
def test_aperiodic_error_tf(ts4sprint_knee, fs, exponent_1, exponent_2):
    irasa_out = irasa_sprint(
        ts4sprint_knee,
        fs=fs,
        band=(0.1, 50),
        overlap_fraction=0.95,
        win_duration=0.5,
        hset_info=(1, 2.0, 0.05),
    )

    irasa_out_bad = irasa_sprint(
        ts4sprint_knee,
        fs=fs,
        band=(0.1, 50),
        overlap_fraction=0.95,
        win_duration=0.5,
        hset_info=(1, 8.0, 0.05),
    )

    kwargs = {
        'cut_spectrum': (1, 40),
        'smooth': True,
        'smoothing_window': 3,
        'min_peak_height': 0.01,
        'peak_width_limits': (0.5, 12),
    }

    assert np.mean(irasa_out.get_aperiodic_error(peak_kwargs=kwargs)) < np.mean(
        irasa_out_bad.get_aperiodic_error(peak_kwargs=kwargs)
    )
