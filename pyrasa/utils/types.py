from typing import Protocol

import numpy as np


class IrasaFun(Protocol):
    def __call__(
        self, data: np.ndarray, fs: int, h: int | None, up_down: str | None, time_orig: np.ndarray | None = None
    ) -> np.ndarray: ...
