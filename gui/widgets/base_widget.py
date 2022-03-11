from typing import Dict, Optional

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import SIGNAL_OWNER


class BaseWidget(qwt.QWidget):

    _signals: Dict[SIGNAL_OWNER, qtc.pyqtSignal]

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        """Base widget class. Contains dictionary of signals
        where one could add signal for any kind of work.
        """
        super().__init__()
        if signals is None:
            self._signals = dict()
        else:
            self._signals = signals

    def enable_widget(self, widget: qwt.QWidget, enabled: bool = True):
        """Enables or disables some widget.

        Parameters
        ----------
        widget : qwt.QWidget
            widget to enable or disable
        enabled : bool, optional
            enable or disable widget, by default True
        """
        widget.setEnabled(enabled)

    @property
    def signals(self) -> Dict[SIGNAL_OWNER, qtc.pyqtSignal]:
        return self._signals
