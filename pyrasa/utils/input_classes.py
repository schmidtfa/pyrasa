from typing import TypedDict


class IrasaSprintKwargsTyped(TypedDict):
    mfft: int
    hop: int
    win_duration: float
    dpss_settings: dict
    win_kwargs: dict
    # smooth: bool
    # n_avgs: list
