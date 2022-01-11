import logging
from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from gui.widgets.common import VWidget, VerticalSpacer

logger = logging.getLogger(__name__)


class InferenceTab(BaseWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
    ):
        super().__init__(signals)
        self._init_ui()

    def _init_ui(self):
        layout = qwt.QHBoxLayout()
        self.setLayout(layout)
        left_part = VWidget()
        layout.addWidget(left_part)
        left_part.setMaximumWidth(300)

        model_gb = qwt.QGroupBox()
        left_part.layout().addWidget(model_gb)
        model_gb.setTitle('Model selection')
        model_gb_layout = qwt.QHBoxLayout(model_gb)

        model_select_btn = qwt.QPushButton(text='select')
        model_gb_layout.addWidget(model_select_btn)
        model_select_btn.clicked.connect(self._load_model)

        left_part.layout().addItem(VerticalSpacer)

        right_part = VWidget()
        layout.addWidget(right_part)

    def _load_model(self) -> None:
        model_path, _ = qwt.QFileDialog.getOpenFileName(
            self,
            'Select model file',
            './models',
            'Models (*.pt)',
        )
        if not model_path:
            logger.warning('No model was selected.')
            return

        logger.debug(f'Selected model path: {model_path}.')
        
