import abc
from dataclasses import dataclass
from typing import Iterator, Tuple

from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam

from core.dataset.configuration import DatasetConfiguration
from core.model.model import DeepfakeModel
from enums import DEVICE
from gui.widgets.preview.configuration import PreviewConfiguration


@dataclass
class OptimizerConfiguration(metaclass=abc.ABCMeta):
    """Class which every optimizer configuration should implement. Provides
    functionality of constructing Optimizer object from the configuration.
    """

    params: Iterator[Parameter]
    learning_rate: float

    @abc.abstractmethod
    def optimizer(self) -> Optimizer:
        """Constructs Optimizer object from the configuration.

        Returns
        -------
        Optimizer
            Optimizer object constructed from options passed to the
            configuration
        """
        ...


@dataclass
class AdamConfiguration(OptimizerConfiguration):

    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0

    def optimizer(self) -> Optimizer:
        return Adam(
            params=self.params,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    def __post_init__(self):
        super().__init__(self.params, self.learning_rate)


@dataclass
class TrainerConfiguration:

    model: DeepfakeModel
    optim_conf: OptimizerConfiguration
    dataset_conf: DatasetConfiguration
    epochs: int
    criterion: _Loss
    device: DEVICE = DEVICE.CPU
    preview_conf: PreviewConfiguration = PreviewConfiguration()
    log_dir: str = 'tensorboard_log'
    checkpoints_dir: str = 'models'
