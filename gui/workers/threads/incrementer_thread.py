from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.worker import IncrementerWorker


class IncrementerThread(WorkerThread):

    def __init__(self, *args, **kwargs):
        super().__init__(IncrementerWorker(), *args, **kwargs)
