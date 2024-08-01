import numpy as np
import pytest
import scipy.signal as dsp

from pyrasa import irasa
from pyrasa.utils.aperiodic_utils import compute_slope
from pyrasa.utils.peak_utils import get_peak_params

from .settings import EXPONENT, FS, MIN_CORR_PSD_CMB, OSC_FREQ, TOLERANCE

# Estimate periodic and aperiodic components with IRASA
# These tests should cover the basic functionality of the IRASA workflow


# fixed slope
@pytest.mark.parametrize('exponent', EXPONENT, scope='session')
@pytest.mark.parametrize('osc_freq', OSC_FREQ, scope='session')
@pytest.mark.parametrize('fs', FS, scope='session')
def test_irasa(combined_signal, fs, osc_freq, exponent):
    f_range = [1, 100]
    irasa_out = irasa(combined_signal, fs, f_range, psd_kwargs={'nperseg': 4 * fs})
    # test the shape of the output
    assert irasa_out.freqs.shape[0] == irasa_out.aperiodic.shape[1] == irasa_out.periodic.shape[1]
    # test the selected frequency range
    assert bool(np.logical_and(irasa_out.freqs[0] == f_range[0], irasa_out.freqs[-1] == f_range[1]))
    # test whether recombining periodic and aperiodic spectrum is equivalent to the original spectrum
    freqs_psd, psd = dsp.welch(combined_signal, fs, nperseg=int(4 * fs))
    psd_cmb = irasa_out.aperiodic[0, :] + irasa_out.periodic[0, :]
    freq_logical = np.logical_and(freqs_psd >= f_range[0], freqs_psd <= f_range[1])
    r = np.corrcoef(psd[freq_logical], psd_cmb)[0, 1]
    assert r > MIN_CORR_PSD_CMB
    # test whether we can reconstruct the exponent correctly
    ap_params, _ = compute_slope(irasa_out.aperiodic, irasa_out.freqs, fit_func='fixed')
    assert bool(np.isclose(ap_params['Exponent'][0], np.abs(exponent), atol=TOLERANCE))
    # test whether we can reconstruct the peak frequency correctly
    pe_params = get_peak_params(irasa_out.periodic, irasa_out.freqs)
    assert bool(np.isclose(np.round(pe_params['cf'], 0), osc_freq))
