from pathlib import Path
from queue import Empty
from threading import Thread
from typing import Optional, Union

import PyQt6.QtCore as qtc
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from core.dataset.dataset import FSDataset
from core.model.configuration import ModelConfig
from core.trainer.fs_trainer import FSTrainer
from core.model.fs import FS
from core.trainer.trainer import StepTrainerConfiguration
from core.worker import Worker
from df_logging.model_logging import DFLogger
from enums import EVENT_DATA_KEY, EVENT_TYPE, JOB_NAME, JOB_TYPE, SIGNAL_OWNER, WIDGET
from message.message import Messages
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
        log_frequency: int,
        sample_frequency: int,
        checkpoint_frequency: int,
        resume: bool,
        resume_run_name: Optional[str] = None,
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
        self._log_freq = log_frequency
        self._sample_freq = sample_frequency
        self._checkpoint_freq = checkpoint_frequency
        self._resume = resume
        if resume:
            self._resume_run_name = resume_run_name
        else:
            self._resume_run_name = None
        self._device = torch.device('cuda')

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
            'train': True,
            'gdeep': self._gdeep,
            'arc_path': r'C:\Users\tonkec\Documents\SimSwap-main\arcface_model\new_arc.tar',
            'lr': self._lr,
            'beta1': self._beta1,
            'lambda_id': self._lambda_id,
            'lambda_feat': self._lambda_feat,
            'lambda_rec': self._lambda_rec,
        }
        model_conf = ModelConfig(
            model=FS,
            options=model_options,
        )
        df_logger = DFLogger(
            model_name=FS.NAME,
            resume_run=self._resume,
            log_frequency=self._log_freq,
            sample_frequency=self._sample_freq,
            checkpoint_frequency=self._checkpoint_freq,
            run_name=self._resume_run_name,
        )
        conf = StepTrainerConfiguration(
            train_data_loader,
            batch_size=self._batch_size,
            steps=self._steps,
            model_config=model_conf,
            df_logger=df_logger,
            resume_run=self._resume,
            device=self._device,
            use_cudnn_benchmark=self._use_cudnn_bench,
        )
        trainer = FSTrainer(conf, self.stop_event)

        conf_wgt_msg = Messages.CONFIGURE_WIDGET(
            SIGNAL_OWNER.FS_TRAINER_WORKER,
            WIDGET.JOB_PROGRESS,
            'setMaximum',
            [self._steps],
            JOB_NAME.TRAIN_FS_MODEL,
        )
        self.send_message(conf_wgt_msg)

        def pf():
            while True:
                try:
                    event = trainer.event_q.get()
                    if event.event_type == EVENT_TYPE.PROGRESS:
                        step = event.data[EVENT_DATA_KEY.PROGRESS_VALUE]
                        if step == -1:
                            break
                        self.report_progress(
                            SIGNAL_OWNER.FS_TRAINER_WORKER,
                            JOB_TYPE.TRAIN_FS_DF_MODEL,
                            step,
                            self._steps,
                        )
                    elif event.event_type == EVENT_TYPE.NEW_SAMPLE:
                        sample = event.data[EVENT_DATA_KEY.SAMPLE_PATH]
                        
                except Empty:
                    ...

        self._pt = Thread(target=pf, daemon=True)
        self._pt.start()

        self.running.emit()
        trainer.start()

    def post_run(self) -> None:
        self._pt.join()
