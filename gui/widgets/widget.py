from typing import Dict

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from enums import SIGNAL_OWNER


class Widget(qwt.QWidget):

    signals: Dict[SIGNAL_OWNER, qtc.pyqtSignal] = dict()

    def __init__(self):
        """Base widget class. Contains dictionary of signals
        where one could add signal for any kind of work.
        """
        super().__init__()

    def add_signal(self, sig: qtc.pyqtSignal, sig_owner: SIGNAL_OWNER):
        """Adds signal to a widget dictionary.

        Parameters
        ----------
        sig : qtc.pyqtSignal
            signal to add
        sig_owner : SIGNAL_OWNER
            widget, worker or some other objects which will to something
            when signal is received
        """
        self.signals[sig_owner] = sig
