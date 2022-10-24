import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torchvision.transforms as T
import wandb

from core.optimizer.configuration import OptimizerConfiguration
from utils import get_date_uid
from variables import APP_LOGGER


logger = logging.getLogger(APP_LOGGER)


class DatasetConfiguration:

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        shuffle: bool = True,
        transforms: T.Compose = T.Compose([T.ToTensor()]),
        num_workers: int = multiprocessing.cpu_count() // 2,
        pin_memory: bool = True,
        drop_last: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        self.root = Path(root)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms,
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

    def values(self) -> Dict[str, Union[str, int, bool]]:
        return {
            'root': str(self.root),
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'pin_memory': self.pin_memory,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'persistent_workers': self.persistent_workers,
        }


class LoggingConfiguration:

    def __init__(
        self,
        model_name: str,
        log_frequency: int,
        sample_frequency: int,
        checkpoint_frequency: int,
        logs_dir: Union[str, Path] = 'logs',
        checkpoints_dir: Union[str, Path] = 'checkpoints',
        samples_dir: Union[str, Path] = 'samples',
        run_name: Optional[str] = None,
        use_wandb: bool = True,
    ) -> None:
        self._model_name = model_name
        self._log_frequency = log_frequency
        self._sample_frequency = sample_frequency
        self._checkpoint_frequency = checkpoint_frequency
        self._use_wandb = use_wandb

        if run_name is None:
            self.run_name = get_date_uid()
        else:
            self.run_name = run_name

        self._logs_dir = Path(logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints_dir = Path(
            checkpoints_dir
        ) / self._model_name / self.run_name
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._samples_dir = Path(
            samples_dir
        ) / self._model_name / self.run_name
        self._samples_dir.mkdir(parents=True, exist_ok=True)

        self._run_log_dir = self._logs_dir / self._model_name / self.run_name
        self._run_log_dir.mkdir(parents=True, exist_ok=True)

        if not use_wandb:
            return

        self._wandb_last_step = 0
        wandb_id_path = self._run_log_dir / 'wandb_id.txt'
        self._wandb_last_step_path = self._run_log_dir / 'wandb_last_step.txt'
        if not (
            wandb_id_path.exists() and self._wandb_last_step_path.exists()
        ):
            wandb_id = wandb.util.generate_id()
            with open(wandb_id_path, 'w+') as f:
                f.write(wandb_id)
            with open(self._wandb_last_step_path, 'w+') as f:
                f.write(str(self._wandb_last_step))
        else:
            with open(wandb_id_path, 'r+') as f:
                wandb_id = f.read()
            with open(self._wandb_last_step_path, 'r+') as f:
                self._wandb_last_step = int(f.read())

        wandb.init(
            dir=str(self._logs_dir),
            project=self._model_name,
            name=self.run_name,
            resume='allow',
            id=wandb_id,
        )

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir

    @property
    def checkpoints_dir(self) -> Path:
        return self._checkpoints_dir

    @property
    def samples_dir(self) -> Path:
        return self._samples_dir

    @property
    def run_log_dir(self) -> Path:
        return self._run_log_dir

    @property
    def log_frequency(self) -> int:
        return self._log_frequency

    @property
    def sample_frequency(self) -> int:
        return self._sample_frequency

    @property
    def checkpoint_frequency(self) -> int:
        return self._checkpoint_frequency

    @property
    def use_wandb(self) -> bool:
        return self._use_wandb

    def values(self) -> Dict[str, Union[str, int]]:
        return {
            'log_dir': str(self._logs_dir),
            'ckeckpoints_dir': str(self._checkpoints_dir),
            'samples_dir': str(self._samples_dir),
            'model_name': self._model_name,
            'log_frequency': self._log_frequency,
            'sample_frequency': self._sample_frequency,
            'checkpoint_frequency': self._checkpoint_frequency,
            'use_wandb': self._use_wandb,
        }

    def update_wandb_last_step(self, step: int) -> None:
        with open(self._wandb_last_step_path, 'w') as f:
            f.write(str(step))
        logger.debug(
            f'Updated last wand step ({step}) in file '
            f'{str(self._wandb_last_step_path)}.'
        )


class ModelConfiguration:

    def __init__(self, args: Dict[str, Any]) -> None:
        self.args = args

    def values(self) -> Dict[str, Any]:
        return self.args


class TrainerConfiguration:

    def __init__(
        self,
        steps: int,
        dataset_conf: DatasetConfiguration,
        logging_conf: LoggingConfiguration,
        optimizer_conf: OptimizerConfiguration,
        model_conf: ModelConfiguration,
        device: torch.device = torch.device('cuda'),
        use_cudnn_benchmark: bool = True,
    ) -> None:
        self.steps = steps
        self.dataset_conf = dataset_conf
        self.logging_conf = logging_conf
        self.optimizer_conf = optimizer_conf
        self.model_conf = model_conf
        self.device = device
        self.use_cudnn_benchmark = use_cudnn_benchmark

    def save(self) -> None:
        d = {
            'steps': self.steps,
            'dataset': self.dataset_conf.values(),
            'logging': self.logging_conf.values(),
            'optimizer': self.optimizer_conf.values(),
            'model': self.model_conf.values(),
            'device': self.device,
            'use_cudnn_benchmark': self.use_cudnn_benchmark,
        }
        print(d)

        # import importlib
        # module = importlib.import_module(d['optimizer']['optimizer_module'])
        # optim_class = getattr(module, d['optimizer']['optimizer_class'])
        # torch.optim.Adam()
        # print(optim_class)
