import logging
from multiprocessing.pool import Pool
from typing import Optional

import PyQt6.QtCore as qtc

from core.worker.worker import Worker


class WorkerWithPool(Worker):
    """Class of worker which uses `multiprocessing.pool.Pool` to split
        work on multiple threads or processes.

        Args:
            num_instances (int): number of threads or processes which will be
                spawned. Defaults to 2.
            message_worker_sig (Optional[qtc.pyqtSignal], optional): signal to
                message worker. Defaults to None.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        num_instances: int = 2,
        message_worker_sig: Optional[qtc.pyqtSignal] = None,
    ) -> None:
        super().__init__(message_worker_sig)

        self._num_instances = num_instances

    def close_pool(self, pool: Pool) -> None:
        """Closes pool of threads or processes which is passed as an argument.
        This method can be used when stop signal is received and the running
        processes or threads need to be terminated.

        Args:
            pool (multiprocessing.pool.Pool): pool which will be closed
        """
        self.logger.debug('Closing process pool.')
        pool.close()
        pool.terminate()
        pool.join()
        self.logger.debug('Process pool closed.')

    def handle_exit(self, pool: Pool) -> None:
        """Gracefully closes pool of processes or threads and emits `finished`
        signal.

        Args:
            pool (multiprocessing.pool.Pool): pool to close on exit
        """
        self.logger.info('Received stop signal, exiting now.')
        self.close_pool(pool)
        self.finished.emit()
