import abc
from typing import Dict, Optional

import PyQt5
import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER

from message.message import Message


class WorkerMeta(PyQt5.sip.wrappertype, abc.ABCMeta):
    ...


class Worker(qtc.QObject, metaclass=WorkerMeta):

    signals = dict()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        *args,
        **kwargs
    ):
        """Base class for any worker object.

        Parameters
        ----------
        signals : Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]], optional
            signals to which worker should have access to, by default dict()
        """
        super().__init__(*args, **kwargs)
        self.signals = signals

    @abc.abstractclassmethod
    @qtc.pyqtSlot(Message)
    def process(self, msg: Message):
        ...
