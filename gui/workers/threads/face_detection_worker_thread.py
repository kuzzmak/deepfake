from gui.workers.threads.worker_thread import WorkerThread
from gui.workers.face_detection_worker import FaceDetectionWorker


class FaceDetectionWorkerThread(WorkerThread):
    def __init__(self, *args, **kwargs):
        super().__init__(FaceDetectionWorker(), *args, **kwargs)
