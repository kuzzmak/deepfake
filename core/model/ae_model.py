import abc

import torch

from core.model.model import DeepfakeModel


class DeepfakeAEModel(DeepfakeModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @abc.abstractmethod
    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder part of the autoencoder network.

        Parameters
        ----------
        x : torch.Tensor
            input for the encoder

        Returns
        -------
        torch.Tensor
            "encoded" input
        """
        ...

    @abc.abstractmethod
    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder part of the autoencoder network.

        Parameters
        ----------
        x : torch.Tensor
            output of the encoder

        Returns
        -------
        torch.Tensor
            decoded input from encoder
        """
        ...
