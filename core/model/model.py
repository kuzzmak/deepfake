import abc

import numpy as np
import torch
import torch.nn as nn


class DeepfakeModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for any Deepfake model. All models should implement this class.
    """

    def __init__(self, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.init_layers()

    @abc.abstractmethod
    def init_layers(self):
        """In this method all layer configuration should be made."""
        ...

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        """Makes one step through the model.

        Parameters
        ----------
        x : torch.Tensor
            input data for first layer of the model
        """
        ...

    def __str__(self):
        """Prints all trainable parameters."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
