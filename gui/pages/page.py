from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget


class Page(QWidget):

    gotoSignal = QtCore.pyqtSignal(str)

    def __init__(self, page_name: str = 'page'):
        super().__init__()

        self.page_name = page_name

    def goto(self, name):
        self.gotoSignal.emit(name)
