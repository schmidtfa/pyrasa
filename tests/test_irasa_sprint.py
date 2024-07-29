import numpy as np
from neurodsp.utils.sim import set_random_seed

from pyrasa.irasa import irasa_sprint
from pyrasa.utils.aperiodic_utils import compute_slope_sprint
from pyrasa.utils.peak_utils import get_band_info, get_peak_params_sprint

from .settings import MIN_R2_SPRINT, TOLERANCE

set_random_seed(42)


# @pytest.mark.parametrize('fs', [(500)], scope='session')
def test_irasa(ts4sprint):
    sgramm_ap, sgramm_p, freqs_ir, times_ir = irasa_sprint(
        ts4sprint[np.newaxis, :], fs=500, band=(1, 100), freq_res=0.5, smooth=False, n_avgs=[3, 7, 11]
    )

    # check basic aperiodic detection
    df_aps, df_gof = compute_slope_sprint(sgramm_ap[np.newaxis, :, :], freqs=freqs_ir, times=times_ir, fit_func='fixed')

    assert df_gof['r_squared'].mean() > MIN_R2_SPRINT
    assert np.isclose(df_aps.query('time < 7')['Exponent'].mean(), 1, atol=TOLERANCE)
    assert np.isclose(df_aps.query('time > 7')['Exponent'].mean(), 2, atol=TOLERANCE)

    # check basic peak detection
    df_peaks = get_peak_params_sprint(
        sgramm_p[np.newaxis, :, :],
        freqs=freqs_ir,
        times=times_ir,
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
