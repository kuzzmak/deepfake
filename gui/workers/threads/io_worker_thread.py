from gui.workers.io_worker import IO_Worker
from gui.workers.threads.worker_thread import WorkerThread


class IO_WorkerThread(WorkerThread):

    def __init__(self, *args, **kwargs):
        super().__init__(IO_Worker(), *args, **kwargs)
