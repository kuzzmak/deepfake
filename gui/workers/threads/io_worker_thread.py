from typing import Dict, Optional

import PyQt6.QtCore as qtc

from enums import SIGNAL_OWNER

from gui.workers.io_worker import IO_Worker
from gui.workers.threads.worker_thread import WorkerThread


class IO_WorkerThread(WorkerThread):
    """Worker for doing any kind of IO operation: saveing, renaming,
    deleting files.

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
            IO_Worker,
            worker_signal,
            signals,
            next_element_signal,
            *args,
            **kwargs
        )
