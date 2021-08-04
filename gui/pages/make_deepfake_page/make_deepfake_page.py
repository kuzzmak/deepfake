from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.pages.make_deepfake_page.data_tab import DataTab
from gui.pages.make_deepfake_page.detection_algorithm_tab \
    import DetectionAlgorithmTab

from message.message import Message

from enums import (
    BODY_KEY,
    DATA_TYPE,
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

    def __init__(self,
                 parent,
                 signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
                 * args,
                 **kwargs):
        super().__init__(parent,
                         signals,
                         page_name=MAKE_DEEPFAKE_PAGE_NAME,
                         *args,
                         **kwargs)

        self.input_data_directory = None
        self.output_data_directory = None

        self.input_data_directory_sig.connect(
            self.input_data_directory_selected)
        self.output_data_directory_sig.connect(
            self.output_data_directory_selected)
        self.extract_frames_sig.connect(self.extract_frames)

        self.init_ui()
        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

    def init_ui(self):
        layout = qwt.QVBoxLayout()

        self.tab_wgt = qwt.QTabWidget()

        data_tab_signals = {
            SIGNAL_OWNER.CONSOLE: self.signals[SIGNAL_OWNER.CONSOLE],
            SIGNAL_OWNER.INPUT_DATA_DIRECTORY: self.input_data_directory_sig,
            SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY: self.output_data_directory_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION: self.extract_frames_sig,
        }
        data_tab = DataTab(data_tab_signals)

        self.tab_wgt.addTab(data_tab, 'Data')
        self.tab_wgt.addTab(DetectionAlgorithmTab(), 'Detection algorithm')

        layout.addWidget(self.tab_wgt)

        self.setLayout(layout)

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
        ...
