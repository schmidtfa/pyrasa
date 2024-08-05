import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa.utils.peak_utils import get_band_info, get_peak_params

from .settings import FS, MANY_OSC_FREQ


# Test slope fitting functionality
@pytest.mark.parametrize('osc_freq', MANY_OSC_FREQ, scope='session')
@pytest.mark.parametrize('fs', FS, scope='session')
def test_peak_detection(oscillation, fs, osc_freq):
    f_range = [1, 250]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(oscillation, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    freqs, psd = freqs[freq_logical], psd[freq_logical]
    # test whether we can reconstruct the peaks correctly
    pe_params = get_peak_params(psd[np.newaxis, :], freqs, min_peak_height=0.1)
    assert bool(np.isclose(pe_params['cf'][0], osc_freq, atol=2))

    pe_filt = get_band_info(pe_params, freq_range=[osc_freq - 2, osc_freq + 2], ch_names=[])
    assert bool(np.isclose(pe_filt['cf'][0], osc_freq, atol=2))


@pytest.mark.parametrize('fs, exponent', [(500, -1)], scope='session')
def test_no_peak_detection(fixed_aperiodic_signal, fs):
    f_range = [1, 250]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(fixed_aperiodic_signal, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    freqs, psd = freqs[freq_logical], psd[freq_logical]
    # test whether we can reconstruct the peaks correctly
    pe_params = get_peak_params(psd[np.newaxis, :], freqs, min_peak_height=0.1)
    assert pe_params.shape[0] == 0


@pytest.mark.parametrize('osc_freq, fs', [(10, 500)], scope='session')
def test_peak_detection_settings(oscillation, fs, osc_freq):
    f_range = [1, 250]
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs, psd = dsp.welch(oscillation, fs, nperseg=int(4 * fs))
    freq_logical = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
    freqs, psd = freqs[freq_logical], psd[freq_logical]
    psd_nan = psd.copy()
    psd_nan[10] = np.nan

    # test if nan in spectrum
    with pytest.raises(ValueError):
        get_peak_params(psd_nan[np.newaxis, :], freqs, min_peak_height=0.1)

    # test if cut spectrum runs
    get_peak_params(psd[np.newaxis, :], freqs, min_peak_height=0.1, cut_spectrum=[1, 40])

    # test if smooth False runs
    get_peak_params(psd[np.newaxis, :], freqs, min_peak_height=0.1, smooth=False)
