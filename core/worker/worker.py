import logging
from multiprocessing import Queue
from multiprocessing.queues import Empty
from typing import Optional, Union

import PyQt6.QtCore as qtc

from enums import (
    BODY_KEY,
    JOB_NAME,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)
from message.message import Body, Message


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

    @property
    def message_worker_sig(self) -> Union[qtc.pyqtSignal, None]:
        return self._message_worker_sig

    def send_message(self, message: Message):
        """Send a message through message worker to wherever needed.

        Parameters
        ----------
        message : Message
            message to send
        """
        if self.message_worker_sig is None:
            return
        self.message_worker_sig.emit(message)

    def should_exit(self) -> bool:
        """Checks if someone requested for this worker to exit.

        Returns
        -------
        bool
            returns `True` if worker should exit, `False` otherwise
        """
        if self._conn_q is None:
            return False
        try:
            _ = self._conn_q.get_nowait()
            return True
        except Empty:
            return False

    @qtc.pyqtSlot()
    def run(self) -> None:
        """Starts the execution of the worker's job, should be called by the
        thread when she starts and not by your explicit call.
        """
        self.started.emit()
        self.run_job()
        self.finished.emit()

    def run_job(self) -> None:
        """Function which contains the code this worker should do.

        Raises
        ------
        NotImplementedError
            raised if not implemented by some worker
        """
        raise NotImplementedError()

    def report_progress(
        self,
        signal_owner: SIGNAL_OWNER,
        job_type: JOB_TYPE,
        part: int,
        total_parts: int,
    ) -> None:
        """Used to report progress to the job widget.

        Parameters
        ----------
        signal_owner : SIGNAL_OWNER
            who sends the message to report progress
        job_type : JOB_TYPE
            what kind of job is executing
        part : int
            current step
        total_parts : int
            total steps
        """
        if self._message_worker_sig is None:
            return
        job_prog_msg = Message(
            MESSAGE_TYPE.ANSWER,
            MESSAGE_STATUS.OK,
            signal_owner,
            SIGNAL_OWNER.JOB_PROGRESS,
            Body(
                job_type,
                {
                    BODY_KEY.PART: part,
                    BODY_KEY.TOTAL: total_parts,
                    BODY_KEY.JOB_NAME: JOB_NAME[job_type.value.upper()].value,
                },
                part == total_parts - 1,
            )
        )
        self.send_message(job_prog_msg)
