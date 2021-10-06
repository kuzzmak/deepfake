from dataclasses import dataclass
from typing import Tuple

from torch.nn.modules.loss import _Loss

from core.dataset.configuration import DatasetConfiguration
from core.model.configuration import ModelConfiguration
from core.optimizer.configuration import OptimizerConfiguration
from enums import DEVICE
from gui.widgets.preview.configuration import PreviewConfiguration


@dataclass
class TrainerConfiguration:
    device: DEVICE
    input_shape: Tuple[int, int, int]
    epochs: int
    criterion: _Loss
    model_conf: ModelConfiguration
    optimizer_conf: OptimizerConfiguration
    dataset_conf: DatasetConfiguration
    preview_conf: PreviewConfiguration
    log_dir: str = 'tensorboard_log'
    checkpoints_dir: str = 'models'
