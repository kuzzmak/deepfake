import abc

import torch.nn as nn

from enums import DEVICE


class ModelFactory(abc.ABCMeta):
    """Base class which every model factory should implement."""

    @abc.abstractstaticmethod
    def build_model(device: DEVICE) -> nn.Module:
        """Builds specific model. Loads weights of the model and
        everything else necessary.

        Parameters
        ----------
        device : DEVICE
            device where computation should be done

        Returns
        -------
        nn.Module
            model
        """
        ...
