from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.message_worker import MessageWorker


class MessageWorkerThread(WorkerThread):

    def __init__(self, *args, **kwargs):
        super().__init__(MessageWorker(), *args, **kwargs)
