import abc
from typing import Optional

import torch
import torch.nn.functional as F


class Activation(metaclass=abc.ABCMeta):
    """Base class which all activation functions should implement.
    """

    def __init__(self, alpha: Optional[float] = 0.):
        """Constructor.

        Parameters
        ----------
        alpha : Optional[float], optional
            constant for the activation function, by default 0.
        """
        self.alpha = alpha

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class LeakyReLu(Activation):
    """LeakyReLu activation function.
    """

    def __init__(self, alpha: float):
        super().__init__(alpha)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, self.alpha)
        return x


class Sigmoid(Activation):
    """Sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        return x
