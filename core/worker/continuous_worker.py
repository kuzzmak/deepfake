from multiprocessing import Queue
from multiprocessing.queues import Empty
import sys
import traceback
from typing import Optional, Union

import PyQt6.QtCore as qtc

from common_structures import Job
from core.worker import Worker


class ContinuousWorker(Worker):

    def __init__(
        self,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._job_q = Queue()
        self._current_job: Union[Job, None] = None

    @property
    def job_q(self) -> Queue:
        return self._job_q

    @qtc.pyqtSlot()
    def run(self) -> None:
        self.started.emit()
        while True:
            try:
                self._current_job = self._job_q.get(timeout=1)
                try:
                    self.run_job()
                except Exception:
                    traceback.print_exc(file=sys.stdout)
                    break
            except Empty:
                if self.should_exit():
                    break
        self.finished.emit()
