from typing import Dict, Optional

import PyQt6.QtCore as qtc

from enums import SIGNAL_OWNER

from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.next_element_worker import NextElementWorker


class NextElementWorkerThread(WorkerThread):
    """Thread for passing signal for next element to some worker.

    Parameters
    ----------
    WorkerThread : WorkerThread
        base worker thread class
    """

    def __init__(
        self,
        worker_signal: qtc.pyqtSignal,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        next_element_signal: Optional[qtc.pyqtSignal] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            NextElementWorker,
            worker_signal,
            signals,
            next_element_signal,
            *args,
            **kwargs
        )
