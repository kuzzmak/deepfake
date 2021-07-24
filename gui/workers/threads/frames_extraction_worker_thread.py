from gui.workers.frames_extraction_worker import FramesExtractionWorker

from gui.workers.threads.worker_thread import WorkerThread


class FramesExtractionWorkerThread(WorkerThread):

    def __init__(self, *args, **kwargs):
        super().__init__(FramesExtractionWorker(), *args, **kwargs)
