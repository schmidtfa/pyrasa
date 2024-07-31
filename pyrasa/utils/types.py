"""Types for the pyrasa package."""

from typing import Protocol

import numpy as np


class IrasaFun(Protocol):
    """Signature for the spectrum generating function needed for IRASA."""

    def __call__(  # noqa: D102
        self, data: np.ndarray, fs: int, h: int | None, up_down: str | None, time_orig: np.ndarray | None = None
    ) -> np.ndarray: ...
