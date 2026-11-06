import numpy as np
import pytest
from neurodsp.sim import sim_peak_oscillation, sim_powerlaw
from numpy.testing import assert_allclose
from scipy.signal import savgol_filter, welch

from pyrasa.utils.irasa_spectrum import IrasaSpectrum
from pyrasa.utils.irasa_tf_spectrum import IrasaTfSpectrum


@pytest.fixture(scope='module')
def simulated_spectrum():
    np.random.seed(42)

    fs = 250
    sig_ap = sim_powerlaw(
        n_seconds=8,
        fs=fs,
        exponent=-1.5,
    )

    sig = sim_peak_oscillation(
        sig_ap,
        fs=fs,
        freq=10,
        bw=2,
        height=1.5,
    )

    freqs, ap = welch(sig_ap, fs=fs, nperseg=500, noverlap=250)
    _, raw = welch(sig, fs=fs, nperseg=500, noverlap=250)

    return freqs, ap, raw - ap


@pytest.mark.parametrize('smooth', [False, True])
def test_pw_rel_known_spectrum(simulated_spectrum, smooth):
    freqs, ap, periodic = simulated_spectrum

    spectrum = IrasaSpectrum(
        freqs=freqs,
        raw_spectrum=(ap + periodic)[None, :],
        aperiodic=ap[None, :],
        periodic=periodic[None, :],
        ch_names=np.array(['A']),
    )

    peaks = spectrum.get_peaks(
        smooth=smooth,
        smoothing_window=2,
        cut_spectrum=(5, 20),
        peak_threshold=0.25,
        peak_width_limits=(1, 8),
    )

    peak = peaks.loc[peaks['pw'].idxmax()]

    assert abs(peak['cf'] - 10) < 1.5  # noqa: PLR2004

    ix = np.argmin(np.abs(freqs - peak['cf']))

    periodic_used = savgol_filter(periodic, 5, 2) if smooth else periodic

    peak_power = max(periodic_used[ix], 0)

    expected = np.log10((ap[ix] + peak_power) / ap[ix])

    assert_allclose(peak['pw_rel'], expected, atol=1e-12)


def test_pw_rel_tf_background_dependence(simulated_spectrum):
    freqs, ap, periodic = simulated_spectrum
    times = np.array([0.1, 0.2])

    # Same oscillatory component, different aperiodic backgrounds
    aperiodic = np.stack([ap, 10 * ap], axis=-1)[None, :, :]
    periodic_tf = np.stack([periodic, periodic], axis=-1)[None, :, :]

    spectrum = IrasaTfSpectrum(
        freqs=freqs,
        time=times,
        aperiodic=aperiodic,
        periodic=periodic_tf,
        raw_spectrum=aperiodic + periodic_tf,
        ch_names=np.array(['A']),
    )

    peaks = spectrum.get_peaks(
        smooth=False,
        cut_spectrum=(5, 20),
        peak_threshold=0.25,
        peak_width_limits=(1, 8),
    )

    peak0 = peaks.loc[peaks['time'] == times[0]].sort_values('pw', ascending=False).iloc[0]

    peak1 = peaks.loc[peaks['time'] == times[1]].sort_values('pw', ascending=False).iloc[0]

    assert_allclose(peak0['cf'], peak1['cf'])
    assert_allclose(peak0['pw'], peak1['pw'])

    # Higher background reduces relative peak power
    assert peak0['pw_rel'] > peak1['pw_rel']
