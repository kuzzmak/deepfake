from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.next_element_worker import NextElementWorker


class NextElementWorkerThread(WorkerThread):

    def __init__(self, *args, **kwargs):
        super().__init__(NextElementWorker(), *args, **kwargs)
