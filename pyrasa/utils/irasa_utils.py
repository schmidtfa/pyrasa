"""Utilities for signal decompositon using IRASA"""

from collections.abc import Callable
from copy import copy

import numpy as np
import scipy.signal as dsp


def _crop_data(
    band: list | tuple, freqs: np.ndarray, psd_aperiodic: np.ndarray, psd_periodic: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Utility function to crop spectra to a defined frequency range"""

    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=axis)
    psd_periodic = np.compress(~mask_freqs, psd_periodic, axis=axis)

    return freqs, psd_aperiodic, psd_periodic


def _gen_time_from_sft(SFT: type[dsp.ShortTimeFFT], sgramm: np.ndarray) -> np.ndarray:  # noqa N803
    """Generates time from SFT object"""

    tmin, tmax = SFT.extent(sgramm.shape[-1])[:2]
    delta_t = SFT.delta_t

    time = np.arange(tmin, tmax, delta_t)
    return time


def _find_nearest(sgramm_ud: np.ndarray, time_array: np.ndarray, time_value: float) -> np.ndarray:
    """Find the nearest time point in an up/downsampled spectrogram"""

    idx = (np.abs(time_array - time_value)).argmin()

    # if sgramm_ud.shape[2] >= idx:
    #     idx = idx
    # elif sgramm_ud.shape[2] <= idx:
    #     idx = sgramm_ud.shape[2] - 1
    if idx < sgramm_ud.shape[2]:
        sgramm_sel = sgramm_ud[:, :, idx]
        return sgramm_sel
    elif idx == sgramm_ud.shape[2]:
        sgramm_sel = sgramm_ud[:, :, idx - 1]
        return sgramm_sel
    else:
        pass


def _get_windows(
    nperseg: int, dpss_settings: dict, win_func: Callable, win_func_kwargs: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a window function used for tapering"""
    low_bias_ratio = 0.9
    max_time_bandwidth = 2.0
    win_func_kwargs = copy(win_func_kwargs)

    # special settings in case multitapering is required
    if win_func == dsp.windows.dpss:
        time_bandwidth = dpss_settings['time_bandwidth']
        if time_bandwidth < max_time_bandwidth:
            raise ValueError(f'time_bandwidth should be >= {max_time_bandwidth} for good tapers')

        n_taps = int(np.floor(time_bandwidth - 1))
        win_func_kwargs.update(
            {
                'NW': time_bandwidth / 2,  # half width
                'Kmax': n_taps,
                'sym': False,
                'return_ratios': True,
            }
        )
        win, ratios = win_func(nperseg, **win_func_kwargs)
        if dpss_settings['low_bias']:
            win = win[ratios > low_bias_ratio]
            ratios = ratios[ratios > low_bias_ratio]
    else:
        win = [win_func(nperseg, **win_func_kwargs)]
        ratios = None

    return win, ratios


def _check_irasa_settings(irasa_params: dict, hset_info: tuple) -> None:
    """Check if the input parameters for irasa are specified correctly"""

    valid_hset_shape = 3
    assert isinstance(irasa_params['data'], np.ndarray), 'Data should be a numpy array.'

    # check if hset is specified correctly
    assert isinstance(
        hset_info, tuple | list | np.ndarray
    ), 'hset should be a tuple, list or numpy array of (min, max, step)'
    assert np.shape(hset_info)[0] == valid_hset_shape, 'shape of hset_info should be 3 i.e. (min, max, step)'

    # check that evaluated range fits with the data settings
    nyquist = irasa_params['fs'] / 2
    hmax = np.max(hset_info)
    band_evaluated: tuple[float, float] = (irasa_params['band'][0] / hmax, irasa_params['band'][1] * hmax)
    assert band_evaluated[0] > 0, 'The evaluated frequency range is 0 or lower this makes no sense'
    assert band_evaluated[1] < nyquist, (
        f'The evaluated frequency range goes up to {np.round(band_evaluated[1], 2)}Hz '
        'which is higher than Nyquist (fs / 2)'
    )

    filter_settings: list[float] = list(irasa_params['filter_settings'])
    if filter_settings[0] is None:
        filter_settings[0] = band_evaluated[0]
    if filter_settings[1] is None:
        filter_settings[1] = band_evaluated[1]

    assert np.logical_and(band_evaluated[0] >= filter_settings[0], band_evaluated[1] <= filter_settings[1]), (
        f'You run IRASA in a frequency range from'
        f'{np.round(band_evaluated[0], irasa_params['hset_accuracy'])} -'
        f'{np.round(band_evaluated[1], irasa_params['hset_accuracy'])}Hz. \n'
        'Your settings specified in "filter_settings" indicate that you have '
        'a bandpass filter from '
        f'{np.round(filter_settings[0], irasa_params['hset_accuracy'])} - '
        f'{np.round(filter_settings[1], irasa_params['hset_accuracy'])}Hz. \n'
        'This means that your evaluated range likely contains filter artifacts. \n'
        'Either change your filter settings, adjust hset or the parameter "band" accordingly. \n'
        f'You want to make sure that band[0] / hset.max() '
        f'> {np.round(filter_settings[0], irasa_params['hset_accuracy'])} '
        f'and that band[1] * hset.max() < {np.round(filter_settings[1], irasa_params['hset_accuracy'])}'
    )
