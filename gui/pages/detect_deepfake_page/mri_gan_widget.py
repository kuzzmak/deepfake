from asyncio.log import logger
import logging
from typing import Dict, Optional

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from core.worker.landmark_extraction_worker import LandmarkExtractionWorker
from enums import CONNECTION, LAYOUT, SIGNAL_OWNER
from gui.pages.detect_deepfake_page.model_widget import ModelWidget
from gui.widgets.common import (
    Button,
    GroupBox,
    HWidget,
    PlayIcon,
    StopIcon,
    VWidget,
    VerticalSpacer,
)
from utils import parse_number


logger = logging.getLogger(__name__)


class MriGanWidget(ModelWidget):

    stop_landmark_extraction_sig = qtc.pyqtSignal()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals)

        self._init_ui()

        self._threads = []
        self._lmrks_extraction_in_progress = False

    def _init_ui(self) -> None:
        lmrks_extraction_gb = GroupBox(
            'Landmark extraction',
            LAYOUT.HORIZONTAL,
        )
        self.data_tab.layout().addWidget(lmrks_extraction_gb)
        lmrks_extraction_gb.setMaximumWidth(400)

        lmrks_extraction_left_part = VWidget()
        lmrks_extraction_gb.layout().addWidget(lmrks_extraction_left_part)

        num_of_proc_row = HWidget()
        lmrks_extraction_left_part.layout().addWidget(num_of_proc_row)
        num_of_proc_row.setMaximumWidth(200)
        num_of_proc_row.layout().setContentsMargins(0, 0, 0, 0)
        num_of_proc_row.layout().addWidget(qwt.QLabel(
            text='number or processes'
        ))

        ext_buttons_row = HWidget()
        lmrks_extraction_left_part.layout().addWidget(ext_buttons_row)
        ext_buttons_row.setMaximumWidth(200)
        ext_buttons_row.layout().setContentsMargins(0, 0, 0, 0)
        data_btn_bg = qwt.QButtonGroup(ext_buttons_row)
        for dt in ['train', 'test', 'valid', 'all']:
            btn = qwt.QRadioButton(dt)
            data_btn_bg.addButton(btn)
            ext_buttons_row.layout().addWidget(btn)

        lmrks_extraction_right_part = VWidget()
        lmrks_extraction_gb.layout().addWidget(lmrks_extraction_right_part)

        self.num_proc_input = qwt.QLineEdit()
        num_of_proc_row.layout().addWidget(self.num_proc_input)
        self.num_proc_input.setText(str(2))

        self.extract_landmarks_btn = Button('start extraction')
        lmrks_extraction_right_part.layout().addWidget(
            self.extract_landmarks_btn
        )
        self.extract_landmarks_btn.setIcon(PlayIcon())
        self.extract_landmarks_btn.clicked.connect(self._extract_landmarks)

        self.configure_ext_lmrks_paths = Button('configure paths')
        lmrks_extraction_right_part.layout().addWidget(
            self.configure_ext_lmrks_paths
        )

        self.data_tab.layout().addItem(VerticalSpacer)

    @qtc.pyqtSlot()
    def _extract_landmarks(self) -> None:
        """Initiates landmark extraction process or stopps it.
        """
        if self._lmrks_extraction_in_progress:
            self._stop_landmark_extraction()
        else:
            num_proc = parse_number(self.num_proc_input.text())
            if num_proc is None:
                logger.error(
                    'Unable to parse your input for number of ' +
                    'processes, you should put integer number.'
                )
                return

            thread = qtc.QThread()
            worker = LandmarkExtractionWorker(
                num_proc,
                self.signals[SIGNAL_OWNER.MESSAGE_WORKER]
            )
            self.stop_landmark_extraction_sig.connect(
                lambda: worker.conn_q.put(CONNECTION.STOP)
            )
            worker.moveToThread(thread)
            self._threads.append((thread, worker))
            thread.started.connect(worker.run)
            worker.started.connect(
                self._on_landmark_extraction_worker_started
            )
            worker.running.connect(
                self._on_landmark_extraction_worker_running
            )
            worker.finished.connect(
                self._on_landmark_extraction_worker_finished
            )
            thread.start()

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_finished(self) -> None:
        """Waits for landmark extraction thread to finish and exit gracefully.
        """
        for thread, _ in self._threads:
            thread.quit()
            thread.wait()
        self._threads = []
        self.enable_widget(self.extract_landmarks_btn, True)
        self.extract_landmarks_btn.setIcon(PlayIcon())
        self.extract_landmarks_btn.setText('start extraction')
        self._lmrks_extraction_in_progress = False

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_started(self) -> None:
        """pyqtSlot which disables button for starting or stopping landmark
        extraction until worker is set up.
        """
        self.enable_widget(self.extract_landmarks_btn, False)
        self._lmrks_extraction_in_progress = True

    @qtc.pyqtSlot()
    def _on_landmark_extraction_worker_running(self) -> None:
        """pyqtSlot which enables button for starting or stopping landmark
        extraction and changes icon of the button to the stop icon.
        """
        self.enable_widget(self.extract_landmarks_btn, True)
        self.extract_landmarks_btn.setIcon(StopIcon())
        self.extract_landmarks_btn.setText('stop extraction')

    def _stop_landmark_extraction(self) -> None:
        logger.info('Requested stop of landmark extraction, please wait...')
        self.enable_widget(self.extract_landmarks_btn, False)
        self.stop_landmark_extraction_sig.emit()
