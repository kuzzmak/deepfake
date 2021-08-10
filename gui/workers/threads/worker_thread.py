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
            so gui doesn't freeze, does some work
        worker_signal : qtc.pyqtSignal
            signal to which someone emits Message for worker to
            do some work
        """
        super().__init__(*args, **kwargs)

        self.worker = worker
        self.worker.moveToThread(self)
        self.worker_signal = worker_signal
        self.worker_signal.connect(self.worker.process)
