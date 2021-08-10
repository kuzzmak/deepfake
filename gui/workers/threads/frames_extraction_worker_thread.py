from typing import Dict, Optional

import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER

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
        next_element_signal: Optional[qtc.pyqtSignal] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            FramesExtractionWorker,
            worker_signal,
            signals,
            next_element_signal,
            *args,
            **kwargs
        )
