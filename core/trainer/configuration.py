import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import wandb

from utils import get_date_uid
from variables import APP_LOGGER


logger = logging.getLogger(APP_LOGGER)


class DatasetConfiguration:

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
    ) -> None:
        self._root = Path(root)
        self._batch_size = batch_size

    def values(self) -> Dict[str, Union[str, int]]:
        return {
            'root': str(self._root),
            'batch_size': self._batch_size,
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

        if run_name is None:
            self._run_name = get_date_uid()
        else:
            self._run_name = run_name

        self._logs_dir = Path(logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints_dir = Path(
            checkpoints_dir
        ) / self._model_name / self._run_name
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._samples_dir = Path(
            samples_dir
        ) / self._model_name / self._run_name
        self._samples_dir.mkdir(parents=True, exist_ok=True)

        self._run_log_dir = self._logs_dir / self._model_name / self._run_name
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
            name=self._run_name,
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

    def values(self) -> Dict[str, Union[str, int]]:
        return {
            'log_dir': str(self._logs_dir),
            'model_name': self._model_name,
            'log_frequency': self._log_freuency,
            'sample_frequency': self._sample_frequency,
            'checkpoint_frequency': self._checkpoint_frequency,
        }

    def update_wandb_last_step(self, step: int) -> None:
        with open(self._wandb_last_step_path, 'w') as f:
            f.write(str(step))
        logger.debug(
            f'Updated last wand step ({step}) in file '
            f'{str(self._wandb_last_step_path)}.'
        )


class OptimizerConfiguration:

    def __init__(
        self,
        optimizer_module: str,
        optimizer_class: str,
        options: Dict[str, Any],
    ) -> None:
        self._optimizer_module = optimizer_module
        self._optimizer_class = optimizer_class
        self._options = options

    def values(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        return {
            'optimizer_module': self._optimizer_module,
            'optimizer_class': self._optimizer_class,
            'options': self._options,
        }


class ModelConfiguration:

    def __init__(self, options: Dict[str, Any]) -> None:
        self._options = options

    def values(self) -> Dict[str, Any]:
        return self._options


class TrainerConfiguration:

    def __init__(
        self,
        dataset_conf: DatasetConfiguration,
        logging_conf: LoggingConfiguration,
        optimizer_conf: OptimizerConfiguration,
        model_conf: ModelConfiguration,
    ) -> None:
        self._dataset_conf = dataset_conf
        self._logging_conf = logging_conf
        self._optimizer_conf = optimizer_conf
        self._model_conf = model_conf

    def save(self) -> None:
        d = {
            'dataset': self._dataset_conf.values(),
            'logging': self._logging_conf.values(),
            'optimizer': self._optimizer_conf.values(),
            'model': self._model_conf.values(),
        }
        print(d)

        # import importlib
        # module = importlib.import_module(d['optimizer']['optimizer_module'])
        # optim_class = getattr(module, d['optimizer']['optimizer_class'])
        # torch.optim.Adam()
        # print(optim_class)
