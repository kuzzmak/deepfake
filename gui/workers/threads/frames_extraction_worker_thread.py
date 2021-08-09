from enums import SIGNAL_OWNER
from typing import Dict, Optional

import PyQt5.QtCore as qtc

from gui.workers.frames_extraction_worker import FramesExtractionWorker
from gui.workers.threads.worker_thread import WorkerThread


class FramesExtractionWorkerThread(WorkerThread):
    """Thread used to split video into single frames.

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
            FramesExtractionWorker(signals),
            worker_signal,
            *args,
            **kwargs
        )
