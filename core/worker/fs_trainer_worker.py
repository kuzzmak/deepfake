from pathlib import Path
from typing import Optional, Union

import PyQt6.QtCore as qtc
from torch.utils.data import DataLoader
import torchvision.transforms as T

from core.dataset.dataset import FSDataset
from core.worker import Worker
from variables import IMAGENET_MEAN, IMAGENET_STD


class FSTrainerWorker(Worker):

    def __init__(
        self,
        steps: int,
        batch_size: int,
        lr: float,
        dataset_root: Union[str, Path],
        gdeep: bool,
        beta1: float,
        lambda_id: float,
        lambda_feat: float,
        lambda_rec: float,
        use_cudnn_benchmark: bool,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._steps = steps
        self._batch_size = batch_size
        self._lr = lr
        self._dataset_root = Path(dataset_root)
        self._gdeep = gdeep
        self._beta1 = beta1
        self._lambda_id = lambda_id
        self._lambda_feat = lambda_feat
        self._lambda_rec = lambda_rec
        self._use_cudnn_bench = use_cudnn_benchmark

    def run_job(self) -> None:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._logger.debug('Constructing FS dataset.')
        dataset = FSDataset(self._dataset_root, transforms)
        train_data_loader = DataLoader(
            dataset=dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        self._logger.debug('Dataset constructed.')
        model_options = {
            'name': 'fs_model',
            'isTrain': True,
            'Gdeep': self._gdeep,
            'arc_path': r'C:\Users\tonkec\Documents\SimSwap-main\arcface_model\arcface2_checkpoint.tar',
            'lr': self._lr,
            'beta1': self._beta1,
            'gpu_ids': 0,
            'checkpoints_dir': 'checkpoints',
            'lambda_id': self._lambda_id,
            'lambda_feat': self._lambda_feat,
            'lambda_rec': self._lambda_rec,
            'log_freq': 10,
            'sample_freq': 10,
            'model_freq': 500,
        }

        self.running.emit()
        self._logger.debug('Started custom job.')