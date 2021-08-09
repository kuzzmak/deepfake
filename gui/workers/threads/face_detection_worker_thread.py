from enums import SIGNAL_OWNER
from typing import Dict, Optional

import PyQt5.QtCore as qtc

from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.face_detection_worker import FaceDetectionWorker


class FaceDetectionWorkerThread(WorkerThread):
    """Thread which detects all faces on input and/or output pictures
    with selected face detection algorithm.

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
            FaceDetectionWorker(signals),
            worker_signal,
            *args,
            **kwargs
        )
