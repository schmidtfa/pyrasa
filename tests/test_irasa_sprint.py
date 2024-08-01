import numpy as np
import pytest
from neurodsp.utils.sim import set_random_seed

from pyrasa.irasa import irasa_sprint
from pyrasa.utils.aperiodic_utils import compute_slope_sprint
from pyrasa.utils.peak_utils import get_band_info, get_peak_params_sprint

from .settings import MIN_R2_SPRINT, TOLERANCE

set_random_seed(42)


def test_irasa_sprint(ts4sprint):
    irasa_tf = irasa_sprint(
        ts4sprint[np.newaxis, :],
        fs=500,
        band=(1, 100),
        freq_res=0.5,
    )

    # check basic aperiodic detection
    slope_fit = compute_slope_sprint(
        irasa_tf.aperiodic[np.newaxis, :, :], freqs=irasa_tf.freqs, times=irasa_tf.time, fit_func='fixed'
    )

    assert slope_fit.gof['r_squared'].mean() > MIN_R2_SPRINT
    assert np.isclose(slope_fit.aperiodic_params.query('time < 7')['Exponent'].mean(), 1, atol=TOLERANCE)
    assert np.isclose(slope_fit.aperiodic_params.query('time > 7')['Exponent'].mean(), 2, atol=TOLERANCE)

    # check basic peak detection
    df_peaks = get_peak_params_sprint(
        irasa_tf.periodic[np.newaxis, :, :],
        freqs=irasa_tf.freqs,
        times=irasa_tf.time,
        smooth=True,
        smoothing_window=1,
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
    assert np.isclose(n_peaks, 8, atol=1)

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
    assert np.isclose(n_peaks, 12, atol=1)


# test settings
def test_irasa_sprint_settings(ts4sprint):
    # test smoothing
    # sgramm_ap, sgramm_p, freqs_ir, times_ir = irasa_sprint(
    #     ts4sprint[np.newaxis, :], fs=500, band=(1, 100), freq_res=0.5, smooth=True, n_avgs=[3]
    # )

    # test dpss
    import scipy.signal as dsp

    irasa_sprint(
        ts4sprint[np.newaxis, :],
        fs=500,
        band=(1, 100),
        win_func=dsp.windows.dpss,
        freq_res=0.5,
        # smooth=False,
        # n_avgs=[3, 7, 11],
    )

    # test too much bandwidht
    with pytest.raises(ValueError):
        irasa_sprint(
            ts4sprint[np.newaxis, :],
            fs=500,
            band=(1, 100),
            win_func=dsp.windows.dpss,
            dpss_settings_time_bandwidth=1,
            freq_res=0.5,
            # smooth=False,
            # n_avgs=[3, 7, 11],
        )
