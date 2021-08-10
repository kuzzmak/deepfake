from typing import Optional, Dict

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import DATA_TYPE, SIGNAL_OWNER

from gui.widgets.base_widget import BaseWidget
from gui.widgets.data_selector import DataSelector


class DataTab(BaseWidget):

    def __init__(self,
                 signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict()):
        super().__init__(signals)

        self.init_ui()

    def init_ui(self):

        layout = qwt.QVBoxLayout()

        center_wgt = qwt.QWidget()
        central_layout = qwt.QHBoxLayout()
        center_wgt.setLayout(central_layout)

        input_wgt_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER],
            SIGNAL_OWNER.INPUT_DATA_DIRECTORY: self.signals[
                SIGNAL_OWNER.INPUT_DATA_DIRECTORY
            ],
            SIGNAL_OWNER.FRAMES_EXTRACTION: self.signals[
                SIGNAL_OWNER.FRAMES_EXTRACTION
            ]
        }
        input_wgt = DataSelector(DATA_TYPE.INPUT, input_wgt_signals)
        central_layout.addWidget(input_wgt)

        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.VLine)
        line.setFrameShadow(qwt.QFrame.Sunken)
        central_layout.addWidget(line)

        output_wgt_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.signals[
                SIGNAL_OWNER.MESSAGE_WORKER
            ],
            SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY: self.signals[
                SIGNAL_OWNER.OUTPUT_DATA_DIRECTORY
            ],
            SIGNAL_OWNER.FRAMES_EXTRACTION: self.signals[
                SIGNAL_OWNER.FRAMES_EXTRACTION
            ]
        }
        output_wgt = DataSelector(DATA_TYPE.OUTPUT, output_wgt_signals)

        central_layout.addWidget(output_wgt)

        layout.addWidget(center_wgt)

        self.setLayout(layout)
