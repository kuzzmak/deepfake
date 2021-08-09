import PyQt5.QtCore as qtc

from gui.workers.worker import Worker


class WorkerThread(qtc.QThread):

    def __init__(self,
                 worker: Worker,
                 worker_signal: qtc.pyqtSignal,
                 *args,
                 **kwargs):
        """Base class for any worker related threads.

        Parameters
        ----------
        worker : Worker
            Worker class which should be in it's separate thread
            so gui doesn't freeze
        """
        super().__init__(*args, **kwargs)

        self.worker = worker
        worker_signal.connect(self.worker.process)
        self.worker.moveToThread(self)
