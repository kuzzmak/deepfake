import abc
from typing import Dict, Optional
import queue

import PyQt5
import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER

from message.message import Message


class WorkerMeta(PyQt5.sip.wrappertype, abc.ABCMeta):
    ...


class Worker(qtc.QObject, metaclass=WorkerMeta):

    signals = dict()
    next_element_sig = qtc.pyqtSignal(bool)
    wait_queue = queue.Queue()

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        wait_queue: Optional[queue.Queue] = None,
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
        self.wait_queue = wait_queue

    def wait_for_element(self):
        while True:
            try:
                elem = self.wait_queue.get(timeout=0.1)
                if elem:
                    return elem
            except queue.Empty:
                ...

    @abc.abstractclassmethod
    @qtc.pyqtSlot(Message)
    def process(self, msg: Message):
        ...
