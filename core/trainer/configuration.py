from __future__ import annotations
from dataclasses import dataclass
import json
import importlib
import inspect
import logging
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torchvision.transforms as T
import wandb

from core.optimizer.configuration import OptimizerConfiguration
from enums import MODEL
from utils import get_date_uid, str_to_bool
from variables import APP_LOGGER


logger = logging.getLogger(APP_LOGGER)


@dataclass
class DatasetConfiguration:
    root: Union[str, Path]
    batch_size: int
    shuffle: bool = True
    transforms: T.Compose = T.Compose([T.ToTensor()])
    num_workers: int = multiprocessing.cpu_count() // 2
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = True

    @classmethod
    def from_dict(
        cls,
        dict: Dict[str, Union[str, int, bool]],
    ) -> DatasetConfiguration:
        transforms_list = dict['transforms']
        transforms = []
        for t in transforms_list:
            t_class = getattr(
                importlib.import_module(t['module']),
                t['class'],
            )
            transform = t_class(**t['args'])
            transforms.append(transform)
        transforms = T.Compose(transforms)

        return cls(
            root=Path(dict['root']), 
            batch_size=dict['batch_size'],
            shuffle=dict['shuffle'],
            num_workers=dict['num_workers'],
            pin_memory=dict['pin_memory'],
            drop_last=dict['drop_last'],
            persistent_workers=dict['persistent_workers'],
            transforms=transforms,
        )

    def values(self) -> Dict[str, Union[str, int, bool]]:
        transforms = []
        if self.transforms != None:
            for transform in self.transforms.transforms:
                transform_module = transform.__class__.__module__
                transform_class_name = transform.__class__.__name__
                transform_class = getattr(
                    importlib.import_module(transform_module),
                    transform_class_name,
                )
                args = [
                    p.name for p in inspect.signature(transform_class) \
                        .parameters.values()
                ]
                transform_dict = {
                    'module': transform_module,
                    'class': transform_class_name,
                    'args': {},
                }
                for arg in args:
                    transform_dict['args'][f'{arg}'] = getattr(transform, arg)
                    
                transforms.append(transform_dict)

        return {
            'root': str(self.root),
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'pin_memory': self.pin_memory,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'persistent_workers': self.persistent_workers,
            'transforms': transforms,
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

        # wandb.init(
        #     dir=str(self._logs_dir),
        #     project=self._model_name,
        #     name=self.run_name,
        #     resume='allow',
        #     id=wandb_id,
        # )

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

    @property
    def wandb_last_step(self) -> int:
        return self._wandb_last_step

    @property
    def latest_checkpoints_file_path(self) -> Path:
        return self._checkpoints_dir / 'latest_checkpoint.txt'

    def values(self) -> Dict[str, Union[str, int]]:
        return {
            'log_dir': str(self._logs_dir),
            'checkpoints_dir': str(self._checkpoints_dir),
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
            f'Updated last wandb step ({step}) in file '
            f'{str(self._wandb_last_step_path)}.'
        )


class ModelConfiguration:

    def __init__(self, model: MODEL, args: Dict[str, Any]) -> None:
        self.model = model
        self.args = args

    def values(self) -> Dict[str, Any]:
        return {
            'model': self.model.value,
            'args': self.args,
        }


class TrainerConfiguration:

    def __init__(
        self,
        steps: int,
        dataset: DatasetConfiguration,
        logging: LoggingConfiguration,
        optimizer: OptimizerConfiguration,
        model: ModelConfiguration,
        resume: bool = False,
        device: torch.device = torch.device('cuda'),
        use_cudnn_benchmark: bool = True,
    ) -> None:
        self.steps = steps
        self.dataset = dataset
        self.logging = logging
        self.optimizer = optimizer
        self.model = model
        self.resume = resume
        self.device = device
        self.use_cudnn_benchmark = use_cudnn_benchmark

    def save(self) -> None:
        d = {
            'steps': self.steps,
            'dataset': self.dataset.values(),
            'logging': self.logging.values(),
            'optimizer': self.optimizer.values(),
            'model': self.model.values(),
            'resume': self.resume,
            'device': str(self.device),
            'use_cudnn_benchmark': self.use_cudnn_benchmark,
        }
        with open(self.logging.run_log_dir / 'configuration.json', 'w') as f:
            f.write(json.dumps(d, indent=4))

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> TrainerConfiguration:
        p = Path(config_path)
        with open(p, 'r') as f:
            obj = json.load(f)
        
        dataset = DatasetConfiguration.from_dict(obj['dataset'])
        print(dataset.root.exists())
        print('dataset', dataset, type(dataset.num_workers))
    #     attributes = [a for a, v in Test.__dict__.items()
    #                   if not re.match('<function.*?>', str(v))
    #                   and not (a.startswith('__') and a.endswith('__'))]
    #     print(attributes)
        # import importlib
        # module = importlib.import_module(d['optimizer']['optimizer_module'])
        # optim_class = getattr(module, d['optimizer']['optimizer_class'])
        # torch.optim.Adam()
        # print(optim_class)
