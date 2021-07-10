import sys

import PyQt5.QtWidgets as qwt

from gui.pages.app_page import AppPage


class App:

    def __init__(self):
        ...

    def gui(self):
        _app = qwt.QApplication(sys.argv)
        gui = AppPage()
        gui.show()
        sys.exit(_app.exec_())
