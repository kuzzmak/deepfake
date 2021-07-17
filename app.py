import sys

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.main_page import MainPage
from gui.workers.io_worker import IO_Worker

from common_structures import IO_OP


class App(qtc.QObject):

    io_op_sig = qtc.pyqtSignal(IO_OP)

    def __init__(self):
        super().__init__()

        self.setup_io_worker()

    def setup_io_worker(self):
        self.io_worker = IO_Worker()
        self.io_worker_thread = qtc.QThread()
        self.io_op_sig.connect(self.io_worker.io_op)
        self.io_worker.moveToThread(self.io_worker_thread)
        self.io_worker_thread.start()

    def gui(self):
        _app = qwt.QApplication(sys.argv)
        gui = MainPage(self)
        gui.show()
        sys.exit(_app.exec_())
