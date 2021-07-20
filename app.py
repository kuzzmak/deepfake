import sys
import json

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.main_page import MainPage
from gui.workers.threads.io_worker_thread import IO_WorkerThread
from gui.workers.threads.message_worker_thread import MessageWorkerThread

from common_structures import IO_OP


class App(qtc.QObject):

    io_op_sig = qtc.pyqtSignal(IO_OP)

    def __init__(self):
        super().__init__()

        self.load_config()

        self.setup_io_worker()
        self.setup_message_worker()

    def setup_io_worker(self):
        self.io_worker_thread = IO_WorkerThread()
        self.io_worker_thread.start()

    def setup_message_worker(self):
        self.message_worker_thread = MessageWorkerThread()
        self.message_worker_thread.start()

    def load_config(self):
        with open('app_config.json') as f:
            self.conf = json.load(f)

    def gui(self):
        _app = qwt.QApplication(sys.argv)
        gui = MainPage(self)
        gui.show()
        sys.exit(_app.exec_())
