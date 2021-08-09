from enums import SIGNAL_OWNER
from typing import Dict, Optional

import PyQt5.QtCore as qtc

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
        *args,
        **kwargs
    ):
        super().__init__(
            IO_Worker(signals),
            worker_signal,
            *args,
            **kwargs
        )
