import logging
from typing import Dict, Optional, Tuple

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from configs.mri_gan_config import MRIGANConfig
from core.worker import TrainMRIGANWorker, Worker
from enums import CONNECTION, JOB_TYPE, LAYOUT, NUMBER_TYPE, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    Button,
    DeviceWidget,
    GroupBox,
    HorizontalSpacer,
    NoMarginLayout,
    PlayIcon,
    StopIcon,
    VerticalSpacer,
)
from utils import parse_number


logger = logging.getLogger(__name__)


class ModelParameter(qwt.QWidget):

    def __init__(self, label: str, config_key: str) -> None:
        super().__init__()

        self._label = label
        self._config_key = config_key

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout(LAYOUT.HORIZONTAL)
        self.setLayout(layout)
        layout.addWidget(qwt.QLabel(text=self._label))
        layout.addItem(HorizontalSpacer())
        self._input = qwt.QLineEdit()
        self._input.setMaximumWidth(100)
        layout.addWidget(self._input)
        self._input.setText(
            str(
                MRIGANConfig
                .get_instance()
                .get_mri_gan_model_params()[self._config_key]
            )
        )

    @property
    def input(self) -> str:
        return self._input.text()


class TrainMRIGANWidget(BaseWidget):
    """Widget containing necessary things to start mri gan training.

    Args:
        signals (Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]], optional):
            dictionary of keys this widget needs. Defaults to None.
    """

    stop_mri_training_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()
        self._train_mri_gan_in_progress = False

        self._init_ui()

    def _init_ui(self) -> None:
        layout = NoMarginLayout()
        self.setLayout(layout)

        gb_mp = GroupBox('Model parameters')
        layout.addWidget(gb_mp)

        self.batch_size = ModelParameter('batch size', 'batch_size')
        gb_mp.layout().addWidget(self.batch_size)

        self.image_size = ModelParameter('image size', 'imsize')
        gb_mp.layout().addWidget(self.image_size)

        self.lr = ModelParameter('lr', 'lr')
        gb_mp.layout().addWidget(self.lr)

        self.epochs = ModelParameter('epochs', 'n_epochs')
        gb_mp.layout().addWidget(self.epochs)

        self.tau = ModelParameter('tau', 'tau')
        gb_mp.layout().addWidget(self.tau)

        self.lambda_pixel = ModelParameter('lambda pixel', 'lambda_pixel')
        gb_mp.layout().addWidget(self.lambda_pixel)

        self._devices = DeviceWidget()
        layout.addWidget(self._devices)

        layout.addItem(VerticalSpacer())

        self.start_training_btn = Button(text='start training')
        layout.addWidget(self.start_training_btn)
        self.start_training_btn.setIcon(PlayIcon())
        self.start_training_btn.clicked.connect(self._mri_gan_train)

        self.setMaximumWidth(250)

    @qtc.pyqtSlot()
    def _mri_gan_train(self) -> None:
        """Initiates or stops training of the mri gan.
        """
        if self._train_mri_gan_in_progress:
            self._stop_mri_gan_training()
            return

        image_size = parse_number(self.image_size.input, NUMBER_TYPE.INT)
        if image_size is None:
            logger.error(
                'Unable to parse image size input, must be integer.'
            )
            return

        batch_size = parse_number(self.batch_size.input, NUMBER_TYPE.INT)
        if batch_size is None:
            logger.error(
                'Unable to parse batch size input, must be integer.'
            )
            return

        lr = parse_number(self.lr.input, NUMBER_TYPE.FLOAT)
        if lr is None:
            logger.error(
                'Unable to parse learning rate input, must be float.'
            )
            return

        epochs = parse_number(self.epochs.input, NUMBER_TYPE.INT)
        if epochs is None:
            logger.error(
                'Unable to parse epochs input, must be integer.'
            )
            return

        thread = qtc.QThread()
        worker = TrainMRIGANWorker(
            image_size,
            batch_size,
            lr,
            epochs,
            self._devices.device,
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
        )
        self.stop_mri_training_sig.connect(
            lambda: worker.conn_q.put(CONNECTION.STOP)
        )
        worker.moveToThread(thread)
        self._threads[JOB_TYPE.TRAIN_MRI_GAN] = (thread, worker)
        thread.started.connect(worker.run)
        worker.started.connect(self._on_train_mri_gan_worker_started)
        worker.running.connect(self._on_train_mri_gan_worker_running)
        worker.finished.connect(self._on_train_mri_gan_worker_finished)
        thread.start()

    def _stop_mri_gan_training(self) -> None:
        """Sends signal to stop training of the mri gan.
        """
        logger.info('Requested stop of the mri gan training, please wait...')
        self.enable_widget(self.start_training_btn, False)
        self.stop_mri_training_sig.emit()

    @qtc.pyqtSlot()
    def _on_train_mri_gan_worker_started(self) -> None:
        """Disables button for starting or stopping process of training mri
        gan until whole setup is done.
        """
        self.enable_widget(self.start_training_btn, False)
        self._train_mri_gan_in_progress = True

    @qtc.pyqtSlot()
    def _on_train_mri_gan_worker_running(self) -> None:
        """Enables button to stop training, chenges text of the button.
        """
        self.enable_widget(self.start_training_btn, True)
        self.start_training_btn.setIcon(StopIcon())
        self.start_training_btn.setText('stop training')

    @qtc.pyqtSlot()
    def _on_train_mri_gan_worker_finished(self) -> None:
        """Gracefully shuts down training thread and changes icon and text of
        the button for starting or stopping training process of the mri gan.
        """
        val = self._threads.get(JOB_TYPE.TRAIN_MRI_GAN, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.TRAIN_MRI_GAN, None)
        self.enable_widget(self.start_training_btn, True)
        self.start_training_btn.setIcon(PlayIcon())
        self.start_training_btn.setText('start training')
        self._train_mri_gan_in_progress = False
