import sys
import json

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.main_page import MainPage

from common_structures import IO_OP


class App(qtc.QObject):

    io_op_sig = qtc.pyqtSignal(IO_OP)

    def __init__(self):
        super().__init__()

        self.load_config()

    def load_config(self):
        with open('app_config.json') as f:
            self.conf = json.load(f)

    def gui(self):
        _app = qwt.QApplication(sys.argv)
        gui = MainPage()
        gui.show()
        sys.exit(_app.exec_())
