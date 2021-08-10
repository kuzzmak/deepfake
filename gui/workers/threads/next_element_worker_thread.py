from typing import Dict, Optional

import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER

from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.next_element_worker import NextElementWorker


class NextElementWorkerThread(WorkerThread):

    def __init__(
        self,
        worker_signal: qtc.pyqtSignal,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        *args,
        **kwargs
    ):
        super().__init__(
            NextElementWorker(signals),
            worker_signal,
            *args,
            **kwargs
        )
