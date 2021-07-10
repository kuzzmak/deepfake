from gui.templates.main_page import Ui_main_page
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt


class Page(qwt.QWidget):

    gotoSignal = qtc.pyqtSignal(str)

    def __init__(self, parent: Ui_main_page, page_name: str = 'page'):
        super().__init__()

        self.parent = parent
        self.page_name = page_name

    def show_menu_bar(self, show):
        self.parent.show_menu_bar_sig.emit(show)

    def show_console(self, show):
        self.parent.show_console_sig.emit(show)

    def print(self, message: str):
        self.parent.console_print_sig.emit(message)

    def goto(self, name):
        self.gotoSignal.emit(name)
