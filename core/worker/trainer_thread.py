import logging
from pathlib import Path
from typing import Iterator, Optional

import PyQt6.QtCore as qtc
import torch
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from core.aligner import Aligner, AlignerConfiguration
from core.dataset.dataset_old import DeepfakeDataset
from core.model.model import DeepfakeModel
from core.model.original_ae import OriginalAE
from core.trainer.configuration import TrainerConfiguration
from core.trainer.trainer import Trainer
from enums import MODEL
from utils import get_aligned_landmarks_filename

logger = logging.getLogger(__name__)


class AlignLandmarksWorker(qtc.QObject):

    def __init__(
        self,
        path_A: Path,
        path_B: Path,
        image_size: int,
        cond: qtc.QWaitCondition,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__()
        self._path_A = path_A
        self._path_B = path_B
        self._image_size = image_size
        self._cond = cond
        self._message_worker_sig = message_worker_sig

    def _align(self, path: Path, person: str) -> None:
        logger.info(f'Aligning landmarks for person {person}.')
        a_c = AlignerConfiguration(
            path,
            self._image_size,
            self._message_worker_sig,
        )
        aligner = Aligner(a_c)
        aligner.align_landmarks()
        logger.info(f'Aligning landmarks for person {person} done.')

    @qtc.pyqtSlot()
    def run(self) -> None:
        try:
            self._align(self._path_A, 'A')
            self._align(self._path_B, 'B')
        finally:
            self._cond.wakeOne()


class TrainingWorker(qtc.QObject):

    finished = qtc.pyqtSignal()

    def __init__(
        self,
        conf: TrainerConfiguration,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__()
        self._conf = conf
        self._message_worker_sig = message_worker_sig
        self._threads = []
        self._mutex = qtc.QMutex()
        self._cond = qtc.QWaitCondition()

    def run(self):
        self._check_for_aligned_landmarks()
        model = self._init_model()
        optimizer = self._init_optimizer(model.parameters())
        data_loader = self._init_data_loader()
        self.trainer = self._init_trainer(model, data_loader, optimizer)
        self.trainer.run()
        self.finished.emit()

    def stop_training(self):
        self.trainer.stop()

    def _check_for_aligned_landmarks(self) -> None:
        landmarks_file = get_aligned_landmarks_filename(
            self._conf.dataset_conf.input_size
        )
        aligned_landmarks_A = self._conf.dataset_conf.path_A / landmarks_file
        aligned_landmarks_B = self._conf.dataset_conf.path_B / landmarks_file
        if aligned_landmarks_A.exists() and aligned_landmarks_B.exists():
            return

        logger.warning(
            f'One or both {landmarks_file} are missing, ' +
            'will align landmarks now.'
        )

        self._mutex.lock()
        self._thread = qtc.QThread()
        self._worker = AlignLandmarksWorker(
            self._conf.dataset_conf.path_A,
            self._conf.dataset_conf.path_B,
            self._conf.dataset_conf.input_size,
            self._cond,
            self._message_worker_sig,
        )
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self._cond.wait(self._mutex)
        self._mutex.unlock()

    def _init_model(self) -> DeepfakeModel:
        logger.info(
            f'Loading model "{self._conf.model_conf.model.value}" to ' +
            f'device "{self._conf.device.value}".'
        )
        model = self._conf.model_conf.model
        input_shape = self._conf.input_shape
        device = self._conf.device
        if model == MODEL.ORIGINAL:
            model = OriginalAE(input_shape)
        model = model.to(device.value)
        logger.info('Model loaded.')
        return model

    def _init_optimizer(self, parameters: Iterator[Parameter]) -> Optimizer:
        optim = self._conf.optimizer_conf.optimizer.value
        optim = getattr(torch.optim, optim)
        optim_args = self._conf.optimizer_conf.optimizer_args
        optim = optim(params=parameters, **optim_args)
        return optim

    def _init_data_loader(self) -> DataLoader:
        conf = self._conf.dataset_conf
        dataset = DeepfakeDataset(
            path_A=conf.path_A,
            path_B=conf.path_B,
            input_size=conf.input_size,
            output_size=conf.output_size,
            transformations=conf.data_transforms,
            image_augmentations=conf.image_augmentations,
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
            criterion=self._conf.criterion,
            device=self._conf.device,
            epochs=self._conf.epochs,
            log_dir=self._conf.log_dir,
            checkpoints_dir=self._conf.checkpoints_dir,
            show_preview=self._conf.preview_conf.show_preview,
            show_preview_comm=self._conf.preview_conf.comm_object,
        )
        return trainer
