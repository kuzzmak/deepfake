from enums import SIGNAL_OWNER
from typing import Dict, Optional

import PyQt5.QtCore as qtc

from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.message_worker import MessageWorker


class MessageWorkerThread(WorkerThread):

    def __init__(
        self,
        worker_signal: qtc.pyqtSignal,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = dict(),
        *args,
        **kwargs
    ):
        super().__init__(
            MessageWorker(signals),
            worker_signal,
            *args,
            **kwargs
        )
