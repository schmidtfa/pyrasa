import numpy as np
from attrs import define


@define
class IrasaSpectrum:
    freqs: np.ndarray
    aperiodic: np.ndarray
    periodic: np.ndarray
