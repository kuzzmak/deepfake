from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.engine import (
    Engine,
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from core.trainer.configuration import TrainerConfiguration


class Trainer:
    """Class that manages everything that has to do with training of the
    particular model.
    """

    def __init__(self, conf: TrainerConfiguration) -> None:
        """Constructor.

        Parameters
        ----------
        conf : TrainerConfiguration
            configuration object for the `Trainer`
        """
        self.model = conf.model
        self.data_loader = conf.dataset_conf.data_loader
        self.optim_conf = conf.optim_conf
        self.criterion = conf.criterion
        self.device = conf.device
        self.epochs = conf.epochs
        self.log_dir = conf.log_dir
        self.checkpoints_dir = conf.checkpoints_dir

    def run(self) -> None:
        """Initiates learning process of the model.
        """
        trainer = create_supervised_trainer(
            self.model,
            self.optim_conf.optimizer(),
            self.criterion,
            device=self.device.value,
        )

        metrics = {"accuracy": Accuracy(), "loss": Loss(self.criterion)}

        train_evaluator = create_supervised_evaluator(
            self.model,
            metrics=metrics,
            device=self.device.value,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine: Engine):
            train_evaluator.run(self.data_loader)

        tb_logger = TensorboardLogger(log_dir=self.log_dir)

        tb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=100),
            tag="training",
            output_transform=lambda loss: {"batchloss": loss},
            metric_names="all",
        )

        def score_function(engine: Engine):
            return engine.state.metrics["accuracy"]

        model_checkpoint = ModelCheckpoint(
            self.log_dir,
            n_saved=2,
            filename_prefix="best",
            score_function=score_function,
            score_name="validation_accuracy",
            global_step_transform=global_step_from_engine(trainer),
            require_empty=False,
        )
        train_evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            model_checkpoint,
            {"model": self.model},
        )

        trainer.run(self.data_loader, max_epochs=self.epochs)

        tb_logger.close()
