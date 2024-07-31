"""Input classes for pyrasa."""

from typing import TypedDict


class IrasaSprintKwargsTyped(TypedDict):
    """TypedDict for the IRASA sprint function."""

    mfft: int
    hop: int
    win_duration: float
    dpss_settings: dict
    win_kwargs: dict
    # smooth: bool
    # n_avgs: list
