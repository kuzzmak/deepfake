import logging
from typing import Dict, Optional

import PyQt6.QtCore as qtc

from core.worker.landmark_extraction_worker import LandmarkExtractionWorker
from enums import CONNECTION, LAYOUT, SIGNAL_OWNER
from gui.pages.detect_deepfake_page.model_widget import ModelWidget
from gui.pages.detect_deepfake_page.mri_gan.common import (
    DataTypeRadioButtons,
    NumOfInstancesRow,
)
from gui.widgets.common import (
    Button,
    GroupBox,
    PlayIcon,
    StopIcon,
    VWidget,
    VerticalSpacer,
)
from gui.widgets.custom_dialog import CustomDialog
from gui.widgets.dialog import Dialog
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
        ##############################
        # LANDMARK EXTRACTION GROUPBOX
        ##############################
        lmrks_extraction_gb = GroupBox(
            'Landmark extraction',
            LAYOUT.HORIZONTAL,
        )
        self.data_tab.layout().addWidget(lmrks_extraction_gb)
        lmrks_extraction_gb.setMaximumWidth(400)

        lmrks_extraction_left_part = VWidget()
        lmrks_extraction_gb.layout().addWidget(lmrks_extraction_left_part)

        self.lmrks_extraction_num_of_instances = NumOfInstancesRow()
        lmrks_extraction_left_part.layout().addWidget(
            self.lmrks_extraction_num_of_instances
        )

        self.ext_lmrks_radio_btns = DataTypeRadioButtons()
        lmrks_extraction_left_part.layout().addWidget(
            self.ext_lmrks_radio_btns
        )

        lmrks_extraction_right_part = VWidget()
        lmrks_extraction_gb.layout().addWidget(lmrks_extraction_right_part)

        self.extract_landmarks_btn = Button('start extraction')
        lmrks_extraction_right_part.layout().addWidget(
            self.extract_landmarks_btn
        )
        self.extract_landmarks_btn.setIcon(PlayIcon())
        self.extract_landmarks_btn.clicked.connect(self._extract_landmarks)

        self.configure_ext_lmrks_paths_btn = Button('configure paths')
        lmrks_extraction_right_part.layout().addWidget(
            self.configure_ext_lmrks_paths_btn
        )
        self.configure_ext_lmrks_paths_btn.clicked.connect(
            self._configure_ext_lmrks_paths
        )

        #########################
        # CROPPING FACES GROUPBOX
        #########################
        crop_faces_gb = GroupBox('Cropping faces', LAYOUT.HORIZONTAL)
        self.data_tab.layout().addWidget(crop_faces_gb)
        crop_faces_gb.setMaximumWidth(400)

        crop_faces_left_part = VWidget()
        crop_faces_gb.layout().addWidget(crop_faces_left_part)

        self.crop_faces_num_of_instances = NumOfInstancesRow()
        crop_faces_left_part.layout().addWidget(
            self.crop_faces_num_of_instances
        )

        self.crop_faces_radio_btns = DataTypeRadioButtons()
        crop_faces_left_part.layout().addWidget(
            self.crop_faces_radio_btns
        )

        crop_faces_right_part = VWidget()
        crop_faces_gb.layout().addWidget(crop_faces_right_part)

        self.crop_faces_btn = Button('crop faces')
        crop_faces_right_part.layout().addWidget(
            self.crop_faces_btn
        )
        self.crop_faces_btn.setIcon(PlayIcon())
        self.crop_faces_btn.clicked.connect(self._crop_faces)

        self.configure_crop_faces_paths = Button('configure paths')
        crop_faces_right_part.layout().addWidget(
            self.configure_crop_faces_paths
        )

        self.data_tab.layout().addItem(VerticalSpacer)

    @qtc.pyqtSlot()
    def _extract_landmarks(self) -> None:
        """Initiates landmark extraction process or stops it.
        """
        if self._lmrks_extraction_in_progress:
            self._stop_landmark_extraction()
        else:
            num_proc = parse_number(
                self.lmrks_extraction_num_of_instances.num_of_instances_value
            )
            if num_proc is None:
                logger.error(
                    'Unable to parse your input for number of ' +
                    'processes, you should put integer number.'
                )
                return

            thread = qtc.QThread()
            worker = LandmarkExtractionWorker(
                self.ext_lmrks_radio_btns.selected_data_type,
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
        """Sends signal to stop landmark extraction.
        """
        logger.info('Requested stop of landmark extraction, please wait...')
        self.enable_widget(self.extract_landmarks_btn, False)
        self.stop_landmark_extraction_sig.emit()

    @qtc.pyqtSlot()
    def _crop_faces(self) -> None:
        ...

    @qtc.pyqtSlot()
    def _configure_ext_lmrks_paths(self) -> None:
        dialog = CustomDialog()
        dialog.exec()
