from typing import Dict, Optional

import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER
from gui.pages.detect_deepfake_page.model_widget import ModelWidget


class MesoNetWidget(ModelWidget):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ) -> None:
        super().__init__(signals)

        self._init_ui()

    def _init_ui(self) -> None:
        ...
