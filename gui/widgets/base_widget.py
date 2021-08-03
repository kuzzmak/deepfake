from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import SIGNAL_OWNER


class BaseWidget(qwt.QWidget):

    _signals: Dict[SIGNAL_OWNER, qtc.pyqtSignal]

    def __init__(self, signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict()):
        """Base widget class. Contains dictionary of signals
        where one could add signal for any kind of work.
        """
        super().__init__()
        self._signals = signals

    @property
    def signals(self) -> Dict[SIGNAL_OWNER, qtc.pyqtSignal]:
        return self._signals
