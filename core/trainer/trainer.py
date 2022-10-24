import logging
from numbers import Number
from queue import Queue
import threading
import time
from typing import Any, Dict, List, Optional

import enlighten
from torch.backends import cudnn
import wandb

from common_structures import Event
from core.trainer.configuration import TrainerConfiguration
from enums import EVENT_DATA_KEY, EVENT_TYPE


class BaseTrainer:

    def __init__(
        self,
        conf: TrainerConfiguration,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self._starting_step = 0
        self._current_step = 0
        self._steps = conf.steps
        self._conf = conf
        self._device = conf.device
        self._train_data_loader = conf.train_data_loader
        self._meters: Dict[str, Number] = {}
        self._log_freq = conf.logging_conf.log_frequency
        self._save_freq = conf.logging_conf.checkpoint_frequency
        self._sample_freq = conf.logging_conf.sample_frequency
        self._checkpoint_dir = conf.logging_conf.checkpoints_dir
        self._samples_dir = conf.logging_conf.samples_dir
        self._use_wandb = self._conf.logging_conf.use_wandb
        cudnn.benchmark = conf.use_cudnn_benchmark
        self._enligten_manager = enlighten.get_manager()
        self._run_name = conf.logging_conf.run_name
        self._logger = logging.getLogger(type(self).__name__)
        if stop_event is not None:
            self._stop_event = stop_event
        else:
            self._stop_event = threading.Event()
        self._event_q: Queue[Event] = Queue()

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
        wandb.config.update(
            {
                'learning_rate': self._conf.model_config.options['lr'],
                'batch_size': self._conf.batch_size,
            }
        )
        wandb.watch(self._model)

    def post_init_logging(self) -> None:
        pass

    @property
    def meters(self) -> Dict[str, Number]:
        return self._meters

    @property
    def event_q(self) -> 'Queue[Event]':
        return self._event_q

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

    def report_progress(self, step: int) -> None:
        self._event_q.put(
            Event(EVENT_TYPE.PROGRESS, {EVENT_DATA_KEY.PROGRESS_VALUE: step})
        )

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
        conf: TrainerConfiguration,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        super().__init__(conf, stop_event)

        self._iters = len(self._train_data_loader)
        self._current_iter = 0
        self._epochs = conf.steps
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
        conf: TrainerConfiguration,
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
