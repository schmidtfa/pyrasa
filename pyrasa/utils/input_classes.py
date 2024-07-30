from typing import TypedDict

import numpy as np


class IrasaKwargsTyped(TypedDict):
    nperseg: int | None
    noverlap: int | None
    nfft: int | None
    h: int | None
    time_orig: None | np.ndarray
    up_down: str | None
    dpss_settings: dict
    win_kwargs: dict


class IrasaSprintKwargsTyped(TypedDict):
    mfft: int
    hop: int
    win_duration: float
    h: int | None
    up_down: str | None
    dpss_settings: dict
    win_kwargs: dict
    time_orig: None | np.ndarray
    # smooth: bool
    # n_avgs: list
