from enums import SIGNAL_OWNER
import PyQt5.QtWidgets as qwt

from gui.widgets.base_widget import BaseWidget
from gui.widgets.data_selector import DataSelector


class DataTab(BaseWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):

        layout = qwt.QVBoxLayout()

        center_wgt = qwt.QWidget()
        central_layout = qwt.QHBoxLayout()
        center_wgt.setLayout(central_layout)

        input_wgt = DataSelector('Input')
        input_wgt.add_signal(
            self.signals[SIGNAL_OWNER.CONOSLE], SIGNAL_OWNER.CONOSLE)
        central_layout.addWidget(input_wgt)

        line = qwt.QFrame()
        line.setFrameShape(qwt.QFrame.VLine)
        line.setFrameShadow(qwt.QFrame.Sunken)
        central_layout.addWidget(line)

        output_wgt = DataSelector('Output')
        output_wgt.add_signal(
            self.signals[SIGNAL_OWNER.CONOSLE], SIGNAL_OWNER.CONOSLE)
        central_layout.addWidget(output_wgt)

        layout.addWidget(center_wgt)

        self.setLayout(layout)
