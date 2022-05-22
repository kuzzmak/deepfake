import logging
from typing import Optional

import PyQt6.QtCore as qtc

from core.worker import ContinuousWorker
from enums import JOB_DATA_KEY

logger = logging.getLogger(__name__)


class InferDFDetectorWorker(ContinuousWorker):

    def __init__(
        self,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

    def run_job(self) -> None:
        self.running.emit()

        path = self._current_job.data.get(JOB_DATA_KEY.IMAGE_PATH, None)
        if path is None:
            logger.error(
                f'Key {JOB_DATA_KEY.IMAGE_PATH.value} must be present.'
            )
            return
        print('received path for job', path)
