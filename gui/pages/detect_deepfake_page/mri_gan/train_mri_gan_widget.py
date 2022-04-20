import logging
from typing import Dict, Optional, Tuple

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from core.worker import TrainMRIGANWorker, Worker
from enums import CONNECTION, JOB_TYPE, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import PlayIcon, StopIcon


logger = logging.getLogger(__name__)


class TrainMRIGANWidget(BaseWidget):

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
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        self.start_training_btn = qwt.QPushButton(text='start training')
        layout.addWidget(self.start_training_btn)
        self.start_training_btn.setIcon(PlayIcon())
        self.start_training_btn.clicked.connect(self._mri_gan_train)

    @qtc.pyqtSlot()
    def _mri_gan_train(self) -> None:
        """Initiates or stops training of the mri gan.
        """
        if self._train_mri_gan_in_progress:
            self._stop_mri_gan_training()
        else:
            thread = qtc.QThread()
            worker = TrainMRIGANWorker(self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ])
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
