import logging
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import enlighten
from ignite.contrib.handlers.tensorboard_logger import (
    GradsScalarHandler,
    TensorboardLogger,
    WeightsScalarHandler,
    # global_step_from_engine,
)
from ignite.engine import (
    Engine,
    Events,
    # create_supervised_evaluator,
)
# from ignite.handlers import ModelCheckpoint
# from ignite.metrics import Loss
from ignite.utils import convert_tensor
import torch
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from common_structures import CommObject
from core.model.model import DeepfakeModel
from enums import DEVICE

logger = logging.getLogger(__name__)


def _prepare_batch(
    batch: Sequence[torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training: pass to a device with options.
    """
    face, target, y = batch
    return (
        convert_tensor(face, device=device, non_blocking=non_blocking),
        convert_tensor(target, device=device, non_blocking=non_blocking),
    )


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable =
    lambda x, y, y_pred, loss: (x, y, y_pred, loss.item()),
) -> Callable:
    """ Function for executing one training step from the `Engine`.

    Parameters
    ----------
    model : torch.nn.Module
        model to train
    optimizer : torch.optim.Optimizer
        optimizer to use
    loss_fn : Union[Callable, torch.nn.Module]
        loss function to use
    device : Optional[Union[str, torch.device]], optional
        device type specification, by default None
    non_blocking : bool, optional
        if True and this copy is between CPU and GPU, the copy may occur
        asynchronously with respect to the host, for other cases, this
        argument has no effect, by default False
    prepare_batch : Callable, optional
        function that receives `batch`, `device`, `non_blocking` and outputs
        tuple of tensors `(batch_x, batch_y)`, by default _prepare_batch
    output_transform : Callable, optional
        function that receives 'x', 'y', 'y_pred', 'loss' and returns value
        to be assigned to engine's state.output after each iteration. Default
        is returning `x, y, y_pred, loss.item()`, by default lambdax

    Returns
    -------
    Callable
        update function
    """
    def _train_step(
        engine: Engine,
        batch: Sequence[torch.Tensor],
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        optimizer.zero_grad()
        face, target = prepare_batch(
            batch,
            device=device,
            non_blocking=non_blocking,
        )
        y_pred = model(face)
        loss = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        return output_transform(face, target, y_pred, loss)

    return _train_step


def _trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable =
    lambda x, y, y_pred, loss: (x, y, y_pred, loss.item()),
) -> Engine:
    trainer = Engine(_training_step(
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking,
        prepare_batch,
        output_transform,
    ))
    return trainer


class Trainer:
    """Class that manages everything that has to do with training of the
    particular model.
    """

    def __init__(
        self,
        model: DeepfakeModel,
        data_loader: DataLoader,
        optimizer: Optimizer,
        criterion: _Loss,
        device: DEVICE,
        epochs: int,
        log_dir: str,
        checkpoints_dir: str,
        show_preview: bool,
        show_preview_comm: CommObject,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.log_dir = log_dir
        self.checkpoints_dir = checkpoints_dir
        self.show_preview = show_preview
        self.show_preview_comm = show_preview_comm

    def _refresh_preview(self, x, y_pred):
        if self.show_preview:
            self.show_preview_comm.data_sig.emit([x, y_pred])

    def run(self) -> None:
        """Initiates learning process of the model.
        """
        n_images = 4

        trainer = _trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.criterion,
            device=self.device.value,
        )

        # progress bars
        manager = enlighten.get_manager()
        epoch_desc = 'Epoch: {}, loss: {:.6f}'
        epoch_pbar = manager.counter(
            total=self.epochs,
            desc=epoch_desc.format(0, 0),
            unit='ticks',
            leave=False,
        )
        iteration_desc = '\tIteration: '
        iteration_pbar = manager.counter(
            total=len(self.data_loader),
            desc=iteration_desc,
            unit='ticks',
            leave=False,
        )

        # metrics = {"loss": Loss(self.criterion)}

        # train_evaluator = create_supervised_evaluator(
        #     self.model,
        #     metrics=metrics,
        #     device=self.device.value,
        # )

        # @trainer.on(Events.EPOCH_COMPLETED)
        # def compute_metrics(engine: Engine):
        #     train_evaluator.run(self.data_loader)

        @trainer.on(Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine: Engine):
            x, y, y_pred, loss = engine.state.output
            epoch_pbar.desc = epoch_desc.format(
                engine.state.epoch,
                loss,
            )
            epoch_pbar.update()
            # reset iteration progress bar
            iteration_pbar.count = 0
            iteration_pbar.start = time.time()

            logger.info(epoch_desc.format(
                engine.state.epoch,
                loss,
            ))
            self._refresh_preview(x[:n_images], y_pred[:n_images])

        @trainer.on(Events.ITERATION_COMPLETED)
        def on_iteration_completed(engine: Engine):
            iteration_pbar.update()

        tb_logger = TensorboardLogger(log_dir=self.log_dir)
        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="training",
            # tuple unpacking is not possible, last element is loss
            output_transform=lambda out: {"batchloss": out[-1]},
            metric_names="all",
        )
        tb_logger.attach(
            trainer,
            log_handler=WeightsScalarHandler(self.model),
            event_name=Events.EPOCH_COMPLETED,
        )
        tb_logger.attach(
            trainer,
            log_handler=GradsScalarHandler(self.model),
            event_name=Events.EPOCH_COMPLETED,
        )

        # def score_function(engine: Engine):
        #     return engine.state.metrics["accuracy"]

        # model_checkpoint = ModelCheckpoint(
        #     self.log_dir,
        #     n_saved=2,
        #     filename_prefix="best",
        #     score_function=score_function,
        #     score_name="validation_accuracy",
        #     global_step_transform=global_step_from_engine(trainer),
        #     require_empty=False,
        # )
        # train_evaluator.add_event_handler(
        #     Events.EPOCH_COMPLETED,
        #     model_checkpoint,
        #     {"model": self.model},
        # )

        trainer.run(self.data_loader, max_epochs=self.epochs)

        tb_logger.close()
        manager.stop()
