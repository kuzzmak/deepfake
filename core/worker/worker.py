import logging
from multiprocessing import Queue
from multiprocessing.queues import Empty
from typing import Optional

import PyQt6.QtCore as qtc

from email.message import Message


class Worker(qtc.QObject):
    """Base class for any worker. Every class that extends this one should
    implement at least `run_job` method which contains all job logic which
    this worker should do. Before this unimplemented method `started` signal
    is emitted and after this method finishes, `finished` signal is emitted.
    You can also, right before the main job starts, emit `running` signal.
    `Worker` also contains `conn_q` property which can be used to stop the job
    of the worker. You simply put ˙CONNECTION.STOP˙ enum into the `Queue`.

    Args:
        message_worker_sig (Optional[qtc.pyqtSignal], optional): signal to the
            message worker if some messages are to be sent. Defaults to None.
    """

    started = qtc.pyqtSignal()
    running = qtc.pyqtSignal()
    finished = qtc.pyqtSignal()

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:

        super().__init__()

        self._message_worker_sig = message_worker_sig
        self._conn_q = Queue()

    @property
    def conn_q(self) -> Queue:
        return self._conn_q

    def send_message(self, message: Message):
        if self._message_worker_sig is not None:
            self._message_worker_sig.emit(message)

    def should_exit(self) -> bool:
        if self._conn_q is None:
            return False
        try:
            _ = self._conn_q.get_nowait()
            return True
        except Empty:
            return False

    @qtc.pyqtSlot()
    def run(self) -> None:
        self.started.emit()
        self.run_job()
        self.finished.emit()

    def run_job(self) -> None:
        raise NotImplementedError()
