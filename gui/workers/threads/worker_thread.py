import PyQt5.QtCore as qtc

from gui.workers.worker import NewWorker


class WorkerThread(qtc.QThread):

    def __init__(self, worker: NewWorker, *args, **kwargs):
        """Base class for any worker related threads.

        Parameters
        ----------
        worker : Worker
            Worker class which should be in it's separate thread
            so gui doesn't freeze
        """
        super().__init__(*args, **kwargs)

        self.worker = worker
        self.worker.moveToThread(self)

    def run(self) -> None:
        return super().run()
