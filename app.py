import sys

import PyQt5.QtWidgets as qwt

from gui.pages.main_page import MainPage


class App:

    def __init__(self):
        ...

    def gui(self):
        _app = qwt.QApplication(sys.argv)
        gui = MainPage()
        gui.show()
        sys.exit(_app.exec_())
