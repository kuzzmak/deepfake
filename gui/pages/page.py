import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt


class Page(qwt.QWidget):

    gotoSignal = qtc.pyqtSignal(str)

    def __init__(self, page_name: str = 'page'):
        super().__init__()

        self.page_name = page_name

    def goto(self, name):
        self.gotoSignal.emit(name)
