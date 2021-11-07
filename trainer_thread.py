import logging
from typing import Iterator

import PyQt5.QtCore as qtc
import torch
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from core.dataset.dataset import DeepfakeDataset
from core.model.model import DeepfakeModel
from core.model.original_ae import OriginalAE
from core.trainer.configuration import TrainerConfiguration
from core.trainer.trainer import Trainer
from enums import MODEL

logger = logging.getLogger(__name__)


class Worker(qtc.QObject):

    finished = qtc.pyqtSignal()

    def __init__(self, conf: TrainerConfiguration):
        super().__init__()
        self.conf = conf

    def run(self):
        model = self._init_model()
        optimizer = self._init_optimizer(model.parameters())
        data_loader = self._init_data_loader()
        self.trainer = self._init_trainer(model, data_loader, optimizer)
        self.trainer.run()
        self.finished.emit()

    def stop_training(self):
        self.trainer.stop()

    def _init_model(self) -> DeepfakeModel:
        logger.info(
            f'Loading model "{self.conf.model_conf.model.value}" to ' +
            f'device "{self.conf.device.value}".'
        )
        model = self.conf.model_conf.model
        input_shape = self.conf.input_shape
        device = self.conf.device
        if model == MODEL.ORIGINAL:
            model = OriginalAE(input_shape).to(device.value)
        model = model.to(device.value)
        logger.debug('Model loaded.')
        return model

    def _init_optimizer(self, parameters: Iterator[Parameter]) -> Optimizer:
        optim = self.conf.optimizer_conf.optimizer.value
        optim = getattr(torch.optim, optim)
        optim_args = self.conf.optimizer_conf.optimizer_args
        optim = optim(params=parameters, **optim_args)
        return optim

    def _init_data_loader(self) -> DataLoader:
        conf = self.conf.dataset_conf
        dataset = DeepfakeDataset(
            metadata_path_A=conf.metadata_path_A,
            metadata_path_B=conf.metadata_path_B,
            input_shape=conf.input_shape,
            image_augmentations=conf.image_augmentations,
            load_into_memory=conf.load_into_memory,
            device=self.conf.device,
            transforms=conf.data_transforms,
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=conf.batch_size,
            shuffle=conf.shuffle,
            # num_workers=self.num_workers,
            num_workers=0,  # > 0 not working on Windows
        )
        return data_loader

    def _init_trainer(
        self,
        model: DeepfakeModel,
        data_loader: DataLoader,
        optimizer: Optimizer,
    ) -> Trainer:
        trainer = Trainer(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=self.conf.criterion,
            device=self.conf.device,
            epochs=self.conf.epochs,
            log_dir=self.conf.log_dir,
            checkpoints_dir=self.conf.checkpoints_dir,
            show_preview=self.conf.preview_conf.show_preview,
            show_preview_comm=self.conf.preview_conf.comm_object,
        )
        return trainer
