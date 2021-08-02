import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.widgets.base_widget import BaseWidget

from message.message import Message


class Page(BaseWidget):

    goto_sig = qtc.pyqtSignal(str)

    def __init__(self, main_page, page_name='page'):
        super().__init__()

        self.page_name = page_name
        self.main_page = main_page

    def send_message(self, msg: Message):
        self.main_page.message_worker_sig.emit(msg)

    def show_menubar(self, show):
        self.main_page.show_menubar_sig.emit(show)

    def show_console(self, show):
        self.main_page.show_console_sig.emit(show)

    def show_toolbar(self, show):
        self.main_page.show_toolbar_sig.emit(show)

    def show_toolbars_and_console(self, show):
        self.show_menubar(show)
        self.show_toolbar(show)
        self.show_console(show)

    def goto(self, name):
        self.goto_sig.emit(name)

    def enable_widget(self, widget: qwt.QWidget, enabled: bool):
        widget.setEnabled(enabled)
