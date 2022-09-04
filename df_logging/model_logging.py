from pathlib import Path
from typing import Optional, Union

import wandb

from utils import get_date_uid


class DFLogger:

    def __init__(
        self,
        model_name: str,
        log_frequency: int,
        sample_frequency: int,
        checkpoint_frequency: int,
        resume_run: bool = False,
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        log_dir: Union[str, Path] = 'logs',
        checkpoints_dir: Union[str, Path] = 'checkpoints',
        samples_dir: Union[str, Path] = 'samples',
    ) -> None:
        self._model_name = model_name
        self._run_name = run_name
        self._log_freq = log_frequency
        self._sample_frequency = sample_frequency
        self._checkpoint_frequency = checkpoint_frequency
        self._resume_run = resume_run
        self._use_wandb = use_wandb
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir = Path(checkpoints_dir)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._samples_dir = Path(samples_dir)
        self._samples_dir.mkdir(parents=True, exist_ok=True)

        if self._run_name is None or not self._resume_run:
            self._run_name = get_date_uid()

        self._proj_log_dir = self._log_dir / self._model_name / self._run_name
        self._proj_log_dir.mkdir(parents=True, exist_ok=True)

        self._wandb_last_step = 0
        if not use_wandb:
            return
        wandb_id_path = self._proj_log_dir / 'wandb_id.txt'
        self._wandb_last_step_path = self._proj_log_dir / 'wandb_last_step.txt'
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
            dir=str(self._log_dir),
            project=self._model_name,
            name=self._run_name,
            resume='allow',
            id=wandb_id,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def log_frequency(self) -> int:
        return self._log_freq

    @property
    def sample_frequency(self) -> int:
        return self._sample_frequency

    @property
    def samples_dir(self) -> Path:
        return self._samples_dir

    @property
    def checkpoint_frequency(self) -> int:
        return self._checkpoint_frequency

    @property
    def use_wandb(self) -> bool:
        return self._use_wandb

    @property
    def log_dir(self) -> Path:
        return self._log_dir.absolute()

    @property
    def checkpoints_dir(self) -> Path:
        return self._checkpoints_dir.absolute()

    @property
    def proj_log_dir(self) -> Path:
        return self._proj_log_dir

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def wandb_last_step(self) -> int:
        return self._wandb_last_step

    @property
    def latest_checkpoints_file_path(self) -> Path:
        return self._checkpoints_dir / 'latest_checkpoint.txt'

    def update_wandb_last_step(self, step: int) -> None:
        with open(self._wandb_last_step_path, 'w+') as f:
            f.write(str(step))
