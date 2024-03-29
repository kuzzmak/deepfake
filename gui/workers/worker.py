import abc
from typing import Dict, Optional
import queue

import PyQt6
import PyQt6.QtCore as qtc

from enums import SIGNAL_OWNER

from message.message import Message

# TODO replace this workers with the new ones


class WorkerMeta(PyQt6.sip.wrappertype, abc.ABCMeta):
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
        """Base worker class.

        Parameters
        ----------
        signals : Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]], optional
            signals to which worker should have access to, by default dict()
        wait_queue : Optional[queue.Queue], optional
            queue to which something should be pushed when another element
            is requested from worker, by default None
        """
        super().__init__(*args, **kwargs)
        self.signals = signals
        self.wait_queue = wait_queue

    def wait_for_element(self):
        while True:
            try:
                elem = self.wait_queue.get(timeout=0.001)
                if elem:
                    return elem
            except queue.Empty:
                ...

    @abc.abstractmethod
    @qtc.pyqtSlot(Message)
    def process(self, msg: Message):
        ...
