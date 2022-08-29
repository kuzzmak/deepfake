import logging
import time
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import enlighten
import torch
import wandb
from torch.backends import cudnn
from torch.nn import Module
from torch.utils.data import DataLoader
from df_logging.model_logging import DFLogger

from utils import get_date_uid


@dataclass
class ModelConfig:
    model: Module
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseTrainerConfiguration:
    train_data_loader: DataLoader
    batch_size: int
    model_config: ModelConfig
    df_logger: DFLogger
    resume_run: bool = False
    device: torch.device = torch.device('cuda')
    use_cudnn_benchmark: bool = False


class EpochIterConfiguration(BaseTrainerConfiguration):

    def __init__(
        self,
        train_data_loader: DataLoader,
        batch_size: int,
        epochs: int,
        model_config: ModelConfig,
        df_logger: DFLogger,
        resume_run: bool = False,
        device: torch.device = torch.device('cuda'),
        use_cudnn_benchmark: bool = False,
    ) -> None:
        super().__init__(
            train_data_loader,
            batch_size,
            model_config,
            df_logger,
            resume_run,
            device,
            use_cudnn_benchmark,
        )

        self._epochs = epochs

    @property
    def epochs(self) -> int:
        return self._epochs


class StepTrainerConfiguration(BaseTrainerConfiguration):

    def __init__(
        self,
        train_data_loader: DataLoader,
        batch_size: int,
        steps: int,
        model_config: ModelConfig,
        df_logger: DFLogger,
        resume_run: bool = False,
        device: torch.device = torch.device('cuda'),
        use_cudnn_benchmark: bool = False,
    ) -> None:
        super().__init__(
            train_data_loader,
            batch_size,
            model_config,
            df_logger,
            resume_run,
            device,
            use_cudnn_benchmark,
        )

        self._steps = steps

    @property
    def steps(self) -> int:
        return self._steps


class BaseTrainer:

    def __init__(
        self,
        conf: BaseTrainerConfiguration,
    ) -> None:
        self._conf = conf
        self._device = conf.device
        self._train_data_loader = conf.train_data_loader

        self._meters: Dict[str, Number] = {}

        self._log_freq = self._conf.df_logger.log_frequency
        self._save_freq = conf.df_logger.checkpoint_frequency
        self._sample_freq = conf.df_logger.sample_frequency
        self._checkpoint_dir = conf.df_logger.checkpoints_dir
        self._samples_dir = conf.df_logger.samples_dir
        self._use_wandb = self._conf.df_logger.use_wandb

        cudnn.benchmark = conf.use_cudnn_benchmark

        self._enligten_manager = enlighten.get_manager()

        self._run_name = conf.df_logger.run_name

        self._logger = logging.getLogger(type(self).__name__)

    def init_model(self) -> None:
        self._logger.info('Loading model.')
        mc = self._conf.model_config
        self._model = mc.model()
        self._model.initialize(mc.options)
        self._logger.info('Model loaded.')

    def init_progress_bars(self) -> None:
        pass

    def post_model_init(self) -> None:
        pass

    def register_meters(self, meters: List[str]) -> None:
        for m in meters:
            self._meters[m] = 0

    def _init_logging(self) -> None:
        self._init_wandb()

    def _init_wandb(self) -> None:
        wandb.config = {
            'learning_rate': self._conf.model_config.options['lr'],
            'batch_size': self._conf.batch_size,
        }
        wandb.watch(self._model)

    def post_init_logging(self) -> None:
        pass

    @property
    def meters(self) -> Dict[str, str]:
        return self._meters

    def update_meter(self, name: str, value: Number) -> None:
        if name not in self._meters:
            raise Exception(f'meter: {name} not registered')
        self._meters[name] = value

    def save_checkpoint(self) -> None:
        pass

    def load_checkpoint(self) -> None:
        pass

    def train(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        self.init_model()
        self.post_model_init()
        if self._conf.resume_run:
            self.load_checkpoint()
        self.init_progress_bars()
        self._init_logging()
        self.post_init_logging()
        try:
            self.train()
        except KeyboardInterrupt:
            print('received stop signal, exiting...')
        finally:
            self._enligten_manager.stop()
            if self._use_wandb:
                self._logger.debug('Closing wandb.')
                wandb.finish()


class EpochIterTrainer(BaseTrainer):

    def __init__(self, conf: EpochIterConfiguration) -> None:
        super().__init__(conf)

        self._iters = len(self._train_data_loader)
        self._current_iter = 0
        self._epochs = conf.epochs
        self._current_epoch = 0

    def init_progress_bars(self) -> None:
        self._epoch_pbar = self._enligten_manager.counter(
            total=self._epochs,
            desc='Epoch',
            unit='ticks',
            leave=False,
        )
        self._iteration_pbar = self._enligten_manager.counter(
            total=len(self._train_data_loader),
            desc='',
            unit='ticks',
            leave=False,
        )

    def _train_one_epoch(self) -> None:
        for batch_idx, data in enumerate(self._train_data_loader):
            self._current_iter = batch_idx
            self.train_one_step(data)

    def train_one_step(self, data) -> None:
        raise NotImplementedError

    def train(self) -> None:
        for e in range(self._epochs):
            self._current_epoch = e
            self._train_one_epoch()
            self._iteration_pbar.count = 0
            self._iteration_pbar.start = time.time()
            self._epoch_pbar.update()


class StepTrainer(BaseTrainer):

    def __init__(self, conf: StepTrainerConfiguration) -> None:
        self._steps = conf.steps

        super().__init__(conf)

        self._starting_step = 0
        self._current_step = 0
        self._train_data_iter = iter(self._train_data_loader)

    def init_progress_bars(self) -> None:
        self._step_pbar = self._init_step_progress_bar(self._starting_step)

    def _init_step_progress_bar(self, start: int = 0):
        return self._enligten_manager.counter(
            count=start,
            total=self._steps,
            unit='ticks',
            leave=False,
            desc='step',
        )

    def get_batch_of_data(self) -> Any:
        try:
            data = next(self._train_data_iter)
        except StopIteration:
            self._train_data_iter = iter(self._train_data_loader)
            data = next(self._train_data_iter)
        return data

    def train_one_step(self) -> None:
        raise NotImplementedError

    def train(self) -> None:
        for s in range(self._starting_step, self._steps):
            self._current_step = s
            self.train_one_step()
            self._step_pbar.update()
            if (self._current_step + 1) % self._save_freq == 0 and \
                    self._current_step > 0:
                self.save_checkpoint()
            if (self._current_step + 1) % self._log_freq == 0:
                if self._use_wandb:
                    self._conf.df_logger.update_wandb_last_step(
                        self._current_step + 1
                    )
                    if self._current_step + 1 > \
                            self._conf.df_logger.wandb_last_step:
                        wandb.log(
                            data=self._meters,
                            step=self._current_step + 1,
                        )
                # TODO log to file
