import logging
from multiprocessing import Queue
from multiprocessing.queues import Empty
from datetime import timedelta
import sys
import threading
import traceback
from typing import Optional, Union

import PyQt6.QtCore as qtc
import enlighten
from enlighten._counter import Counter

from enums import (
    BODY_KEY,
    JOB_TYPE,
    MESSAGE_STATUS,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
)
from message.message import Body, Message, Messages
from utils import format_timedelta


class Worker(qtc.QObject):
    """Base class for any worker. Every class that extends this one should
    implement at least `run_job` method which contains all job logic which
    this worker should do. Before this unimplemented method `started` signal
    is emitted and after this method finishes, `finished` signal is emitted.
    You can also, right before the main job starts, emit `running` signal.
    `Worker` also contains `conn_q` property which can be used to stop the job
    of the worker. You simply put ˙CONNECTION.STOP˙ enum into the `Queue`.
    If worker needs to output some data, that can be done by connecting to
    the `output` signal.

    Args:
        message_worker_sig (Optional[qtc.pyqtSignal], optional): signal to the
            message worker if some messages are to be sent. Defaults to None.
    """

    started = qtc.pyqtSignal()
    running = qtc.pyqtSignal()
    finished = qtc.pyqtSignal()
    output = qtc.pyqtSignal(dict)


    def __init__(
        self,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:

        super().__init__()

        self._message_worker_sig = message_worker_sig
        self._conn_q = Queue()
        self._tick_manager = enlighten.get_manager()
        self._ticks = None
        self._forced_exit = False
        self._logger = logging.getLogger(type(self).__name__)
        self._stop_event = threading.Event()

    @property
    def conn_q(self) -> Queue:
        return self._conn_q

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def message_worker_sig(self) -> Union[qtc.pyqtSignal, None]:
        return self._message_worker_sig

    @property
    def ticks(self) -> Union[Counter, None]:
        return self._ticks

    def send_message(self, message: Message):
        """Send a message through message worker to wherever needed.

        Parameters
        ----------
        message : Message
            message to send
        """
        if self.message_worker_sig is None:
            return
        # intercept message and set new ticks if the message is to
        # configure job progress widget before the job starts
        if message.recipient == SIGNAL_OWNER.CONFIGURE_WIDGET and \
                message.body.data[BODY_KEY.METHOD] == 'setMaximum':
            self._init_ticks(message.body.data[BODY_KEY.ARGS][0])
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

    def post_run(self) -> None:
        pass

    @qtc.pyqtSlot()
    def run(self) -> None:
        """Starts the execution of the worker's job, should be called by the
        thread when she starts and not by your explicit call.
        """
        self.started.emit()
        try:
            self.run_job()
        except Exception:
            traceback.print_exc(file=sys.stdout)
        finally:
            self.finished.emit()
            self.send_message(Messages.JOB_EXIT())
            self._ticks = None
            self.post_run()

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
                    BODY_KEY.ETA: self._calculate_eta(),
                },
                part == total_parts - 1,
            )
        )
        self.send_message(job_prog_msg)

    def _init_ticks(self, length: int) -> None:
        """Initializes counter which tracks how much until job is finished.

        Parameters
        ----------
        length : int
            number of steps
        """
        self._ticks = self._tick_manager.counter(total=length)

    def _calculate_eta(self) -> Union[str, None]:
        """Calculates how much is left till the job is done and formats the
        remaining time in convenient format.

        Returns
        -------
        Union[str, None]
            remaining time if the ticks were initialized
        """
        if self.ticks is None:
            return None
        self.ticks.update()
        iterations = self.ticks.count - self.ticks.start_count
        elapsed = self.ticks.elapsed
        rate = (iterations / elapsed) if elapsed else 0
        eta = round((self.ticks.total - iterations) / rate, 2)
        eta = timedelta(seconds=eta)
        return format_timedelta(eta)

    def stop(self) -> None:
        self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()
