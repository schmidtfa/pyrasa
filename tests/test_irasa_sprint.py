import numpy as np
import pytest
from neurodsp.utils.sim import set_random_seed

from pyrasa.irasa import irasa_sprint
from pyrasa.utils.peak_utils import get_band_info

from .settings import EXPONENT, FS, MIN_R2_SPRINT, SPRINT_TOLERANCE

set_random_seed(42)


@pytest.mark.parametrize('fs', FS, scope='session')
@pytest.mark.parametrize('exponent_1', EXPONENT, scope='session')
@pytest.mark.parametrize('exponent_2', EXPONENT, scope='session')
def test_irasa_sprint(ts4sprint, fs, exponent_1, exponent_2):
    irasa_tf = irasa_sprint(
        ts4sprint[np.newaxis, :],
        win_duration=0.5,
        overlap_fraction=0.98,
        fs=fs,
        band=(0.1, 50),
        hset_info=(1.05, 4.0, 0.05),
    )

    # check basic aperiodic detection
    slope_fit = irasa_tf.fit_aperiodic_model(fit_func='fixed')
    #       irasa_tf.aperiodic[np.newaxis, :, :], freqs=irasa_tf.freqs, times=irasa_tf.time,
    #  )

    assert slope_fit.gof['R2'].mean() > MIN_R2_SPRINT
    assert np.isclose(
        np.mean(slope_fit.aperiodic_params.query('time < 7')['Exponent']), np.abs(exponent_1), atol=SPRINT_TOLERANCE
    )
    assert np.isclose(
        np.mean(slope_fit.aperiodic_params.query('time > 7')['Exponent']), np.abs(exponent_2), atol=SPRINT_TOLERANCE
    )

    # check basic peak detection
    df_peaks = irasa_tf.get_peaks(
        cut_spectrum=(1, 40),
        smooth=True,
        smoothing_window=3,
        min_peak_height=0.01,
        peak_width_limits=(0.5, 12),
    )

    df_alpha = get_band_info(df_peaks, freq_range=(8, 12), ch_names=[])
    alpha_peaks = df_alpha.query('pw > 0.10')
    t_diff = 0.025

    alpha_ts = alpha_peaks['time'].to_numpy()
    n_peaks = 0
    for ix, i in enumerate(alpha_ts):
        try:
            diff = alpha_ts[ix + 1] - i
            if diff > t_diff:
                n_peaks += 1
        except IndexError:
            pass

    # one missing burst is ok for now
    assert np.isclose(n_peaks, 7, atol=4)

    df_beta = get_band_info(df_peaks, freq_range=(20, 30), ch_names=[])
    beta_peaks = df_beta.query('pw > 0.10')

    beta_ts = beta_peaks['time'].to_numpy()
    n_peaks = 0
    for ix, i in enumerate(beta_ts):
        try:
            diff = beta_ts[ix + 1] - i
            if diff > t_diff:
                n_peaks += 1
        except IndexError:
            pass

    # one missing burst is ok for now
    assert np.isclose(n_peaks, 11, atol=4)


# test settings
@pytest.mark.parametrize('fs', [1000], scope='session')
@pytest.mark.parametrize('exponent_1', [-1], scope='session')
@pytest.mark.parametrize('exponent_2', [-2], scope='session')
def test_irasa_sprint_settings(ts4sprint, fs):
    # test dpss
    import scipy.signal as dsp

    irasa_sprint(
        ts4sprint[np.newaxis, :],
        fs=fs,
        band=(0.1, 100),
        win_func=dsp.windows.dpss,
    )

    # test too much bandwidht
    with pytest.raises(ValueError):
        irasa_sprint(
            ts4sprint[np.newaxis, :],
            fs=fs,
            band=(1, 100),
            win_func=dsp.windows.dpss,
            dpss_settings_time_bandwidth=4,
        )

    # test ratios
    with pytest.raises(ValueError):
        irasa_sprint(
            ts4sprint[np.newaxis, :],
            fs=fs,
            band=(1, 100),
            win_func=dsp.windows.dpss,
            dpss_settings_time_bandwidth=4,
        )
