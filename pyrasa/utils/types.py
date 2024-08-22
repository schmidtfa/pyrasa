"""Custom classes for pyrasa."""

from typing import Protocol, TypedDict

import numpy as np
import pandas as pd
from attrs import define


class IrasaFun(Protocol):
    """
    A protocol defining the interface for an IRASA function used in the PyRASA library.

    The `IrasaFun` protocol specifies the expected signature of a function used to separate
    aperiodic and periodic components of a power spectrum using the IRASA algorithm.
    Any function conforming to this protocol can be passed to other PyRASA functions
    as a custom IRASA implementation.

    Methods
    -------
    __call__(data: np.ndarray, fs: int, h: float,
             up_down: str | None, time_orig: np.ndarray | None = None) -> np.ndarray
        Separates the input data into its aperiodic and periodic components based on the IRASA method.

    Parameters
    ----------
    data : np.ndarray
        The input time series data to be analyzed.
    fs : int
        The sampling frequency of the input data.
    h : float
        The resampling factor used in the IRASA algorithm.
    up_down : str | None
        A string indicating the direction of resampling ('up' or 'down').
        If None, no resampling is performed.
    time_orig : np.ndarray | None, optional
        The original time points of the data, used for interpolation if necessary.
        If None, no interpolation is performed.

    Returns
    -------
    np.ndarray
        The output of the IRASA function.
    """

    def __call__(
        self, data: np.ndarray, fs: int, h: float, up_down: str | None, time_orig: np.ndarray | None = None
    ) -> np.ndarray: ...


class IrasaSprintKwargsTyped(TypedDict):
    """TypedDict for the IRASA sprint function."""

    nfft: int
    hop: int
    win_duration: float
    dpss_settings: dict
    win_kwargs: dict


@define
class AperiodicFit:
    """Container for the results of aperiodic model fits."""

    aperiodic_params: pd.DataFrame
    gof: pd.DataFrame
    model: pd.DataFrame
