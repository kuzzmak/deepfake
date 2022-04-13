import sys

import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from gui.pages.main_page import MainPage


class App(qtc.QObject):

    def __init__(self):
        super().__init__()

    def gui(self):
        _app = qwt.QApplication(sys.argv)
        gui = MainPage()
        _app.aboutToQuit.connect(gui.terminate_threads)
        gui.show()
        sys.exit(_app.exec())
