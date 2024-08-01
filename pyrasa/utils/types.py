from typing import Protocol, TypedDict

import numpy as np
import pandas as pd
from attrs import define


class IrasaFun(Protocol):
    def __call__(
        self, data: np.ndarray, fs: int, h: int | None, up_down: str | None, time_orig: np.ndarray | None = None
    ) -> np.ndarray: ...


class FitFun(Protocol):
    def __call__(self, x: np.ndarray, *args: float, **kwargs: float) -> np.ndarray: ...


class IrasaSprintKwargsTyped(TypedDict):
    mfft: int
    hop: int
    win_duration: float
    dpss_settings: dict
    win_kwargs: dict


@define
class SlopeFit:
    aperiodic_params: pd.DataFrame
    gof: pd.DataFrame
