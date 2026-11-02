import numpy as np
import pytest
from scipy.signal.windows import dpss

from pyrasa.utils.irasa_utils import _get_windows


@pytest.mark.parametrize(
    'time_bandwidth, expected_tapers',
    [(2.0, 1), (3.0, 2), (4.0, 3), (6.0, 5)],
)
def test_dpss_taper_count(time_bandwidth, expected_tapers):
    settings = {
        'time_bandwidth': time_bandwidth,
        'low_bias': True,
        'eigenvalue_weighting': True,
    }

    windows, ratios = _get_windows(
        nperseg=256,
        dpss_settings=settings,
        win_func=dpss,
        win_func_kwargs={},
    )

    assert windows.shape == (expected_tapers, 256)
    assert len(ratios) == expected_tapers
    assert np.all(ratios > 0.9)  # noqa: PLR2004


def test_dpss_without_eigenvalue_weighting():
    settings = {
        'time_bandwidth': 4.0,
        'low_bias': True,
        'eigenvalue_weighting': False,
    }

    windows, ratios = _get_windows(
        nperseg=256,
        dpss_settings=settings,
        win_func=dpss,
        win_func_kwargs={},
    )

    assert windows.shape == (3, 256)
    assert ratios is None


def test_dpss_invalid_time_bandwidth():
    settings = {
        'time_bandwidth': 1.5,
        'low_bias': True,
        'eigenvalue_weighting': True,
    }

    with pytest.raises(ValueError, match='time_bandwidth'):
        _get_windows(
            nperseg=256,
            dpss_settings=settings,
            win_func=dpss,
            win_func_kwargs={},
        )
