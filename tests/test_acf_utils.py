import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrasa.utils.acf_utils import IrasaACF, psd_to_acf
from pyrasa.utils.irasa_spectrum import IrasaSpectrum


@pytest.mark.parametrize('start', [0.0, 0.25, 5.0])
def test_matches_independent_cosine_sum(start):
    freq = np.arange(start, 200.000001, 0.25)
    power = (1 + (2 * np.pi * freq * 0.02) ** 2) ** -1

    lags, corrs = psd_to_acf(freq, power, max_lag=0.08, lag_step=0.001)

    expected = (
        np.sum(
            power[:, None] * np.cos(2 * np.pi * freq[:, None] * lags),
            axis=0,
        )
        / power.sum()
    )

    assert_allclose(corrs, expected, rtol=1e-11, atol=1e-11)
    assert corrs[0] == pytest.approx(1.0)


def test_channels_independent_and_normalized():
    freqs = np.arange(0, 500.25, 0.25)
    tau1, tau2 = 0.015, 0.060

    psd = np.vstack([1 / (1 + (2 * np.pi * freqs * tau) ** 2) for tau in (tau1, tau2)])

    # Convert to the one-sided Welch PSD convention
    psd[:, 1:-1] *= 2

    lags, corrs = psd_to_acf(freqs, psd, max_lag=0.2)

    assert corrs.shape == (2, lags.size)
    assert_allclose(corrs[:, 0], 1.0)

    result = IrasaACF(lags, corrs)
    half = result.half_life()

    assert_allclose(
        half,
        np.array([tau1, tau2]) * np.log(2),
        atol=0.002,
    )


def test_missing_half_life_returns_nan():
    obj = IrasaACF(np.array([0.0, 0.01, 0.02]), np.array([[1.0, 0.8, 0.6]]))
    assert np.isnan(obj.half_life()[0])


def test_out_of_range_lag_rejected():
    acf = IrasaACF(np.array([0.0, 0.01]), np.ones((2, 2)))
    with pytest.raises(ValueError, match='delay'):
        acf.at_lag(0.02)


def test_magnitude_scaling_invariance():
    freqs = np.arange(0.25, 200.25, 0.25)
    power = 1 / (1 + (2 * np.pi * freqs * 0.02) ** 2)
    _, reference = psd_to_acf(freqs, power)
    for factor in (1e-280, 1e280):
        _, actual = psd_to_acf(freqs, factor * power)
        assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_to_acf_method_with_pyrasa_attributes():
    freqs = np.arange(0.25, 200.25, 0.25)
    aperiodic = np.ones((2, freqs.size))
    raw_spectrum = aperiodic * np.array([[1.0], [3.0]])

    spectrum = IrasaSpectrum(
        freqs=freqs,
        aperiodic=aperiodic,
        raw_spectrum=raw_spectrum,
        periodic=raw_spectrum - aperiodic,
        ch_names=np.array(['A', 'B']),
    )

    acf = spectrum.to_acf(
        source='aperiodic',
        max_lag=0.05,
    )

    assert list(acf.ch_names) == ['A', 'B']
    assert acf.source == 'aperiodic'
    assert_allclose(acf.at_lag(0), [1, 1])

    with pytest.raises(ValueError, match='source'):
        spectrum.to_acf(source='periodic')


def test_to_acf_source_selection():
    freqs = np.arange(0.25, 200.25, 0.25)

    aperiodic = 1 / (1 + (2 * np.pi * freqs * 0.02) ** 2)
    raw = aperiodic + np.exp(-0.5 * ((freqs - 10) / 2) ** 2)

    spectrum = IrasaSpectrum(
        freqs=freqs,
        aperiodic=aperiodic[np.newaxis, :],
        raw_spectrum=raw[np.newaxis, :],
        periodic=(raw - aperiodic)[np.newaxis, :],
        ch_names=np.array(['A']),
    )

    acf_ap = spectrum.to_acf(source='aperiodic')
    acf_raw = spectrum.to_acf(source='raw')

    assert not np.allclose(acf_ap.corrs, acf_raw.corrs)
