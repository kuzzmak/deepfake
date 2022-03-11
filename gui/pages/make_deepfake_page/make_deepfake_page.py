from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt
from gui.pages.make_deepfake_page.inference_tab import InferenceTab
from gui.pages.page import Page
from gui.pages.make_deepfake_page.data_tab import DataTab
from gui.pages.make_deepfake_page.detection_algorithm_tab \
    import DetectionAlgorithmTab
from gui.pages.make_deepfake_page.training_tab import TrainingTab
from message.message import Body, Message
from enums import (
    BODY_KEY,
    DATA_TYPE,
    DEVICE,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)
from names import (
    MAKE_DEEPFAKE_PAGE_NAME,
    MAKE_DEEPFAKE_PAGE_TITLE,
)


class MakeDeepfakePage(Page):

    input_data_directory_sig = qtc.pyqtSignal(str)
    output_data_directory_sig = qtc.pyqtSignal(str)
    extract_frames_sig = qtc.pyqtSignal(Message)
    detect_faces_sig = qtc.pyqtSignal(Message)

    def __init__(
        self,
        parent,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent,
            signals,
            page_name=MAKE_DEEPFAKE_PAGE_NAME,
            *args,
            **kwargs,
        )

        self.input_data_directory = None
        self.output_data_directory = None

        self.input_data_directory_sig.connect(
            self.input_data_directory_selected,
        )
        self.output_data_directory_sig.connect(
            self.output_data_directory_selected,
        )
        self.extract_frames_sig.connect(self.extract_frames)
        self.detect_faces_sig.connect(self.detect_faces)

        self.init_ui()
        self.add_signals()
        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

    def init_ui(self):
        layout = qwt.QVBoxLayout()

        self.tab_wgt = qwt.QTabWidget()

        ind = data_tab_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ],
            SIGNAL_OWNER.INPUT_DATA_DIRECTORY: self.input_data_directory_sig,
            SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY: self.output_data_directory_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION: self.extract_frames_sig,
        }
        data_tab = DataTab(data_tab_signals)
        self.tab_wgt.addTab(data_tab, 'Data')

        detection_tab_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ]
        }
        detection_tab = DetectionAlgorithmTab(
            detection_tab_signals
        )
        self.tab_wgt.addTab(detection_tab, 'Detection')

        training_tab_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ]
        }
        training_tab = TrainingTab(training_tab_signals)
        self.tab_wgt.addTab(training_tab, 'Training')

        inference_tab_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ]
        }
        inference_tab = InferenceTab(inference_tab_signals)
        self.tab_wgt.addTab(inference_tab, 'Inference')
        self.tab_wgt.setCurrentIndex(ind)
        layout.addWidget(self.tab_wgt)
        self.setLayout(layout)

    def add_signals(self):
        msg = Message(
            MESSAGE_TYPE.REQUEST,
            MESSAGE_STATUS.OK,
            SIGNAL_OWNER.MAKE_DEEPFAKE_PAGE,
            SIGNAL_OWNER.MESSAGE_WORKER,
            Body(
                JOB_TYPE.ADD_SIGNAL,
                {
                    BODY_KEY.SIGNAL_OWNER:
                    SIGNAL_OWNER.MAKE_DEEPFAKE_PAGE_DETECT_FACES,
                    BODY_KEY.SIGNAL: self.detect_faces_sig,
                }
            )
        )
        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)

    @qtc.pyqtSlot(str)
    def input_data_directory_selected(self, directory: str):
        self.input_data_directory = directory

    @qtc.pyqtSlot(str)
    def output_data_directory_selected(self, directory: str):
        self.output_data_directory = directory

    def progress_value_changed(self, value: int):
        if value == 100:
            msg = qwt.QMessageBox(self)
            msg.setIcon(qwt.QMessageBox.Information)
            msg.setText("Face extraction successful.")
            msg.setInformativeText("Extracted faces are shown below.")
            msg.setWindowTitle("Face extraction information")
            msg.setStandardButtons(qwt.QMessageBox.Ok)
            msg.exec_()

    @qtc.pyqtSlot(Message)
    def extract_frames(self, msg: Message):
        """Starts process of extracting frames from video.
        """
        if msg.body.data[BODY_KEY.DATA_TYPE] == DATA_TYPE.INPUT:
            msg.body.data[BODY_KEY.DATA_DIRECTORY] = self.input_data_directory
        else:
            msg.body.data[BODY_KEY.DATA_DIRECTORY] = self.output_data_directory

        msg.sender = SIGNAL_OWNER.FRAMES_EXTRACTION
        msg.recipient = SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER

        self.send_message(msg)

    def detect_faces(self, msg: Message):
        """Receives signal from detection algorithm tab which contains
        faces directories, algorithm and model path. Adds input and output
        directories and forwards received message to face detection worker.

        Parameters
        ----------
        msg : Message
            message from detection algorithm tab
        """
        msg.body.data[BODY_KEY.INPUT_DATA_DIRECTORY] = \
            self.input_data_directory
        msg.body.data[BODY_KEY.OUTPUT_DATA_DIRECTORY] = \
            self.output_data_directory
        msg.sender = SIGNAL_OWNER.MAKE_DEEPFAKE_PAGE
        msg.recipient = SIGNAL_OWNER.FACE_DETECTION_WORKER

        self.signals[SIGNAL_OWNER.MESSAGE_WORKER].emit(msg)
