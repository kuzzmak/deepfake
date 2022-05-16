import logging
from typing import Dict, Optional, Tuple

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from core.worker import TrainDeepfakeDetectorWorker, Worker
from enums import JOB_TYPE, SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import (
    Button,
    DeviceWidget,
    GroupBox,
    PlayIcon,
    StopIcon,
    VerticalSpacer,
)

logger = logging.getLogger(__name__)


class TrainDeepfakeDetectorWidget(BaseWidget):
    """Widget responsible for the configuration and running the df
    detector training.

    Parameters
    ----------
    signals : Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]], optional
        signals for this widget, by default None
    """

    stop_df_detector_training_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals)

        self._threads: Dict[JOB_TYPE, Tuple[qtc.QThread, Worker]] = dict()
        self._train_df_detector_in_progress = False

        self._init_ui()

    def _init_ui(self):
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        gb = GroupBox('Model parameters')
        layout.addWidget(gb)

        self.devices = DeviceWidget()
        layout.addWidget(self.devices)

        layout.addItem(VerticalSpacer())

        self.start_training_btn = Button('start training')
        layout.addWidget(self.start_training_btn)
        self.start_training_btn.setIcon(PlayIcon())
        self.start_training_btn.clicked.connect(self._train_df_detector)

    @qtc.pyqtSlot()
    def _train_df_detector(self) -> None:
        """Initiates or stops training of the df detector.
        """
        if self._train_df_detector_in_progress:
            self._stop_df_detector_training()
            return
        thread = qtc.QThread()
        worker = TrainDeepfakeDetectorWorker(
            self.devices.device,
            self.signals[SIGNAL_OWNER.MESSAGE_WORKER],
        )
        worker.moveToThread(thread)
        self._threads[JOB_TYPE.TRAIN_DF_DETECTOR] = (thread, worker)
        thread.started.connect(worker.run)
        worker.started.connect(self._on_train_df_detector_worker_started)
        worker.running.connect(self._on_train_df_detector_worker_running)
        worker.finished.connect(self._on_train_df_detector_worker_finished)
        thread.start()

    def _stop_df_detector_training(self) -> None:
        """Sends signal to stop training of the df detector.
        """
        logger.info(
            'Requested stop of the df detector training, ' +
            'please wait...'
        )
        self.enable_widget(self.start_training_btn, False)
        self.stop_df_detector_training_sig.emit()

    @qtc.pyqtSlot()
    def _on_train_df_detector_worker_started(self) -> None:
        """Disables button for starting or stopping process of training df
        detector until whole setup is done.
        """
        self.enable_widget(self.start_training_btn, False)
        self._train_df_detector_in_progress = True

    @qtc.pyqtSlot()
    def _on_train_df_detector_worker_running(self) -> None:
        """Enables button to stop training, chenges text of the button.
        """
        self.enable_widget(self.start_training_btn, True)
        self.start_training_btn.setIcon(StopIcon())
        self.start_training_btn.setText('stop training')

    @qtc.pyqtSlot()
    def _on_train_df_detector_worker_finished(self) -> None:
        """Gracefully shuts down training thread and changes icon and text of
        the button for starting or stopping training process of the df
        detector.
        """
        val = self._threads.get(JOB_TYPE.TRAIN_DF_DETECTOR, None)
        if val is not None:
            thread, _ = val
            thread.quit()
            thread.wait()
            self._threads.pop(JOB_TYPE.TRAIN_DF_DETECTOR, None)
        self.enable_widget(self.start_training_btn, True)
        self.start_training_btn.setIcon(PlayIcon())
        self.start_training_btn.setText('start training')
        self._train_df_detector_in_progress = False
