import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa.utils.peak_utils import get_peak_params

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
    # test whether we can reconstruct the exponent correctly
    pe_params = get_peak_params(psd[np.newaxis, :], freqs, min_peak_height=0.1)
    assert bool(np.isclose(pe_params['cf'][0], osc_freq, atol=2))
