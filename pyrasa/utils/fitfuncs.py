import abc
import numpy as np


class AbstractFitFun(abc.ABC):
    def __init__(self, *args: Any, **kwargs: Any):
        pass

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, *args: float, **kwargs: float) -> np.ndarray:
        pass

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        return {}


class FixedFitFun(AbstractFitFun):
    def __init__(self, x: np.ndarray):
        self.x = x

    def __call__(self, x: np.ndarray, b0: float, b: float, *args: float, **kwargs: float) -> np.ndarray:
        y_hat = b0 - np.log10(x ** b)

        return y_hat

    @property
    def curve_kwargs(self) -> dict[str, Any]:
        return {"b0": 0.0, "b": 0.0}
