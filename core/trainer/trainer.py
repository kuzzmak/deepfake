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
    project: str = 'trainer'
    run_name: Optional[str] = None
    resume_run: bool = False
    device: torch.device = torch.device('cpu')
    logging_dir: Union[str, Path] = 'logs'
    log_freq: Optional[int] = None
    use_cudnn_benchmark: bool = False
    wandb: bool = False


class EpochIterConfiguration(BaseTrainerConfiguration):

    def __init__(
        self,
        train_data_loader: DataLoader,
        batch_size: int,
        epochs: int,
        model_config: ModelConfig,
        project: str = 'trainer',
        run_name: Optional[str] = None,
        resume_run: bool = False,
        device: torch.device = torch.device('cpu'),
        logging_dir: Union[str, Path] = 'logs',
        log_freq: Optional[int] = None,
        use_cudnn_benchmark: bool = False,
        wandb: bool = False,
    ) -> None:
        super().__init__(
            train_data_loader,
            batch_size,
            model_config,
            project,
            run_name,
            resume_run,
            device,
            logging_dir,
            log_freq,
            use_cudnn_benchmark,
            wandb,
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
        project: str = 'trainer',
        run_name: Optional[str] = None,
        resume_run: bool = False,
        device: torch.device = torch.device('cpu'),
        logging_dir: Union[str, Path] = 'logs',
        log_freq: Optional[int] = None,
        use_cudnn_benchmark: bool = False,
        wandb: bool = False,
    ) -> None:
        super().__init__(
            train_data_loader,
            batch_size,
            model_config,
            project,
            run_name,
            resume_run,
            device,
            logging_dir,
            log_freq,
            use_cudnn_benchmark,
            wandb,
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
        self._logging_dir = Path(conf.logging_dir)
        self._train_data_loader = conf.train_data_loader

        self._meters: Dict[str, Number] = {}

        self._checkpoint_dir = Path(
            self._conf.model_config.options['checkpoints_dir']
        )
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_freq = self._conf.model_config.options['model_freq']
        self._sample_freq = self._conf.model_config.options['sample_freq']

        cudnn.benchmark = conf.use_cudnn_benchmark

        self._enligten_manager = enlighten.get_manager()

        self._run_name = self._conf.run_name
        if self._run_name is None:
            self._run_name = get_date_uid()

    def init_model(self) -> None:
        mc = self._conf.model_config
        self._model = mc.model()
        self._model.initialize(mc.options)

    def init_progress_bars(self) -> None:
        pass

    def post_model_init(self) -> None:
        pass

    def register_meters(self, meters: List[str]) -> None:
        for m in meters:
            self._meters[m] = 0

    def _init_logging(self) -> None:
        self._sample_path: Path = self._checkpoint_dir / \
            self._conf.model_config.options['name'] / 'samples'
        self._sample_path.mkdir(parents=True, exist_ok=True)

        self._proj_log_dir = self._logging_dir / self._conf.project / self._run_name
        self._proj_log_dir.mkdir(parents=True, exist_ok=True)

        self._init_wandb()

    def _init_wandb(self) -> None:
        if not self._conf.wandb:
            return

        wandb_id_path = self._proj_log_dir / 'wandb_id.txt'
        self._wandb_last_step_path = self._proj_log_dir / 'wandb_last_step.txt'
        if not (
            wandb_id_path.exists() and self._wandb_last_step_path.exists()
        ):
            wandb_id = wandb.util.generate_id()
            with open(wandb_id_path, 'w+') as f:
                f.write(wandb_id)
            self._wandb_last_step = 0
            with open(self._wandb_last_step_path, 'w+') as f:
                f.write(str(self._wandb_last_step))
        else:
            with open(wandb_id_path, 'r+') as f:
                wandb_id = f.read()
            with open(self._wandb_last_step_path, 'r+') as f:
                self._wandb_last_step = int(f.read())

        wandb.init(
            dir=str(self._conf.logging_dir),
            project=self._conf.project,
            name=self._run_name,
            resume='allow',
            id=wandb_id,
        )
        wandb.config = {
            'learning_rate': self._conf.model_config.options['lr'],
            'batch_size': self._conf.batch_size,
        }
        wandb.watch(self._model)

    def update_wandb_last_step(self, step: int) -> None:
        with open(self._wandb_last_step_path, 'w+') as f:
            f.write(str(step))

    def post_init_logging(self) -> None:
        pass

    @property
    def meters(self) -> Dict[str, str]:
        return self._meters

    def update_meter(self, name: str, value: Number) -> None:
        if name not in self._meters:
            raise Exception(f'meter: {name} not registered')
        self._meters[name] = value

    def get_latest_checkpoints_file_path(self) -> Path:
        return self._checkpoint_dir / 'latest_checkpoint.txt'

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
            if self._conf.wandb:
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
            if (self._current_step + 1) % self._conf.log_freq == 0:
                if self._conf.wandb:
                    self.update_wandb_last_step(self._current_step + 1)
                    if self._current_step + 1 > self._wandb_last_step:
                        wandb.log(
                            data=self._meters,
                            step=self._current_step + 1,
                        )
                # TODO log to file
