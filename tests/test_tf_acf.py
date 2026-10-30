import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrasa.utils.acf_utils import IrasaACF, psd_to_acf
from pyrasa.utils.irasa_tf_spectrum import IrasaTfSpectrum


@pytest.fixture
def tf_spectrum():
    freqs = np.arange(0.25, 200.25, 0.25)
    times = np.array([0.1, 0.2, 0.3])

    # Different timescales across channels and time windows
    taus = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
        ]
    )

    aperiodic = 1 / (1 + (2 * np.pi * freqs[None, :, None] * taus[:, None, :]) ** 2)

    periodic = np.exp(-0.5 * ((freqs - 10) / 2) ** 2)[None, :, None] * np.ones_like(aperiodic)

    return IrasaTfSpectrum(
        freqs=freqs,
        time=times,
        aperiodic=aperiodic,
        periodic=periodic,
        raw_spectrum=aperiodic + periodic,
        ch_names=np.array(['A', 'B']),
    )


def test_tf_acf_shapes(tf_spectrum):
    acf = tf_spectrum.to_acf(max_lag=0.1)

    assert acf.corrs.shape == (2, 3, len(acf.lags))
    assert acf.half_life().shape == (2, 3)
    assert acf.at_lag(0.036).shape == (2, 3)

    assert_allclose(acf.corrs[..., 0], 1.0)
    assert_allclose(acf.time, tf_spectrum.time)
    assert list(acf.ch_names) == ['A', 'B']


def test_tf_acf_matches_individual_windows(tf_spectrum):
    acf = tf_spectrum.to_acf(max_lag=0.1)

    for ix in range(len(tf_spectrum.time)):
        lags, corrs = psd_to_acf(
            tf_spectrum.freqs,
            tf_spectrum.aperiodic[:, :, ix],
            max_lag=0.1,
            lag_step=0.001,
        )

        assert_allclose(acf.lags, lags)
        assert_allclose(acf.corrs[:, ix, :], corrs)

        # Compare the features against stationary ACF results
        single_acf = IrasaACF(lags=lags, corrs=corrs)

        assert_allclose(
            acf.half_life()[:, ix],
            single_acf.half_life(),
        )

        assert_allclose(
            acf.at_lag(0.036)[:, ix],
            single_acf.at_lag(0.036),
        )


def test_tf_acf_source_selection(tf_spectrum):
    acf_ap = tf_spectrum.to_acf(source='aperiodic')
    acf_raw = tf_spectrum.to_acf(source='raw')

    assert not np.allclose(acf_ap.corrs, acf_raw.corrs)
    assert acf_ap.source == 'aperiodic'
    assert acf_raw.source == 'raw'

    with pytest.raises(ValueError, match='source'):
        tf_spectrum.to_acf(source='periodic')


def test_tf_acf_single_channel(tf_spectrum):
    spectrum = IrasaTfSpectrum(
        freqs=tf_spectrum.freqs,
        time=tf_spectrum.time,
        aperiodic=tf_spectrum.aperiodic[0],
        raw_spectrum=tf_spectrum.raw_spectrum[0],
        periodic=tf_spectrum.periodic[0],
        ch_names=np.array(['A']),
    )

    acf = spectrum.to_acf(max_lag=0.1)

    assert acf.corrs.shape == (1, 3, len(acf.lags))
    assert acf.half_life().shape == (1, 3)
    assert_allclose(acf.at_lag(0), np.ones((1, 3)))
