import logging
from queue import Queue
import threading
import time
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Dict, List, Optional

import enlighten
import torch
import wandb
from torch.backends import cudnn
from torch.nn import Module
from torch.utils.data import DataLoader

from df_logging.model_logging import DFLogger


@dataclass
class ModelConfig:
    model: Module
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseTrainerConfiguration:
    train_data_loader: DataLoader
    batch_size: int
    steps: int
    model_config: ModelConfig
    df_logger: DFLogger
    resume_run: bool
    device: torch.device
    use_cudnn_benchmark: bool
    name: str

    def __str__(self) -> str:
        val = f'{self.name} TRAINER CONFIGURATION\n'
        val += '--------------------------\n'
        val += f'batch size:          {self.batch_size}\n'
        val += f'steps:               {self.steps}\n'
        val += f'resume_run:          {self.resume_run}\n'
        val += f'use_cudnn_benchmark: {self.use_cudnn_benchmark}\n'
        val += f'device:              {self.device}\n'
        val += f'model options:\n'
        longest_key = max([len(k) for k in self.model_config.options.keys()])
        for k, v in self.model_config.options.items():
            v = str(v).rjust(longest_key - len(k) + len(str(v)))
            val += f'{k}: {v}\n'
        return val


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
            'EPOCH',
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
            steps,
            model_config,
            df_logger,
            resume_run,
            device,
            use_cudnn_benchmark,
            'STEP',
        )


class BaseTrainer:

    def __init__(
        self,
        conf: BaseTrainerConfiguration,
        stop_event: Optional[threading.Event],
    ) -> None:
        self._starting_step = 0
        self._current_step = 0
        self._steps = conf.steps
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
        if stop_event is not None:
            self._stop_event = stop_event
        else:
            self._stop_event = threading.Event()
        self._progress_q = Queue()

    def stop(self) -> None:
        self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

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

    def post_training(self) -> None:
        pass

    def report_progress(self) -> None:
        self._progress_q.put({})

    def log(self) -> None:
        if (self._current_step + 1) % self._log_freq == 0 and self._use_wandb:
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

    def start(self) -> None:
        self.init_model()
        self.post_model_init()
        if self._conf.resume_run:
            self.load_checkpoint()
        self.init_progress_bars()
        self._init_logging()
        self.post_init_logging()
        try:
            self._logger.info('Training started.')
            self.train()
        except KeyboardInterrupt:
            print('Received stop signal, exiting...')
        finally:
            self._enligten_manager.stop()
            if self._use_wandb:
                self._logger.debug('Closing wandb.')
                wandb.finish()
            self.post_training()
            self._logger.info('Training finished.')


class EpochIterTrainer(BaseTrainer):

    def __init__(
        self,
        conf: EpochIterConfiguration,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__(conf, stop_event)

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

    def __init__(
        self,
        conf: StepTrainerConfiguration,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__(conf, stop_event)
        
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
            if self.should_stop():
                self._logger.info('Requested stop, please wait...')
                break
            self._current_step = s
            self.train_one_step()
            self._step_pbar.update()
            if (self._current_step + 1) % self._save_freq == 0 and \
                    self._current_step > 0:
                self.save_checkpoint()
            self.log()
