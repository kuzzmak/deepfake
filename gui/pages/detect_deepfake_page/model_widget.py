from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import VWidget


class ModelWidget(BaseWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals)

        self._init_main_ui()

    def _init_main_ui(self) -> None:
        layout = qwt.QVBoxLayout()
        self.setLayout(layout)

        self.tab_wgt = qwt.QTabWidget()
        layout.addWidget(self.tab_wgt)

        self.data_tab = VWidget()
        self.tab_wgt.addTab(self.data_tab, 'Data')

        self.training_tab = VWidget()
        self.tab_wgt.addTab(self.training_tab, 'Training')

        self.inference_tab = VWidget()
        self.tab_wgt.addTab(self.inference_tab, 'Inference')
