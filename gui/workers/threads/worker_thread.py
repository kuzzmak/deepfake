import queue
from typing import Dict, Optional

import PyQt6.QtCore as qtc

from enums import SIGNAL_OWNER

from gui.workers.worker import Worker


class WorkerThread(qtc.QThread):

    def __init__(
        self,
        worker: Worker,
        worker_signal: qtc.pyqtSignal,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        next_element_signal: Optional[qtc.pyqtSignal] = None,
        *args,
        **kwargs,
    ):
        """Base class for any worker related threads.

        Parameters
        ----------
        worker : Worker
            Worker class which should be in it's separate thread
            so gui doesn't freeze, does some work
        worker_signal : qtc.pyqtSignal
            signal to which someone emits Message for worker to
            do some work
        """
        super().__init__(*args, **kwargs)

        if next_element_signal is not None:
            self.w_q = queue.Queue()
            self.worker = worker(signals, self.w_q)
            next_element_signal.connect(self.next_element)
        else:
            self.worker = worker(signals)

        self.worker.moveToThread(self)
        self.worker_signal = worker_signal
        self.worker_signal.connect(self.worker.process)

    @qtc.pyqtSlot()
    def next_element(self):
        self.w_q.put(1)
