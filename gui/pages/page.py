from enum import Enum
from collections import namedtuple

import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from constants import CONSOLE_TEXT_SIZE

from enums import IO_OP_TYPE

ConsolePrefix = namedtuple('ConsolePrefix', 'prefix prefix_color')

console_message_template = '<span style="font-size:{}pt; color:{}; white-space:pre;">{}<span>'


class CONSOLE_COLORS(Enum):
    RED = '#ff0000'
    BLACK = '#000000'
    ORANGE = '#ffa500'


class CONSOLE_MESSAGE_TYPE(Enum):
    LOG = ConsolePrefix('[LOG]', CONSOLE_COLORS.BLACK)
    INFO = ConsolePrefix('[INFO]', CONSOLE_COLORS.BLACK)
    ERROR = ConsolePrefix('[ERROR]', CONSOLE_COLORS.RED)
    WARNING = ConsolePrefix('[WARNING]', CONSOLE_COLORS.ORANGE)


class Page(qwt.QWidget):

    gotoSignal = qtc.pyqtSignal(str)

    def __init__(self, main_page, page_name='page'):
        super().__init__()

        self.page_name = page_name
        self.app = main_page.app
        self.main_page = main_page

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

    def _print(self, message: str):
        self.main_page.console_print_sig.emit(message)

    def _get_console_message_prefix(self, message_type: CONSOLE_MESSAGE_TYPE):
        prefix_color = message_type.value.prefix_color.value
        prefix = message_type.value.prefix
        prefix = console_message_template.format(
            CONSOLE_TEXT_SIZE, prefix_color, f'{prefix: <11}')
        return prefix

    def print(self, message: str, message_type: CONSOLE_MESSAGE_TYPE):
        prefix = self._get_console_message_prefix(message_type)
        text = prefix + \
            console_message_template.format(
                CONSOLE_TEXT_SIZE, CONSOLE_COLORS.BLACK.value, message)
        self._print(text)

    def goto(self, name):
        self.gotoSignal.emit(name)

    def enable_widget(self, widget: qwt.QWidget, enabled: bool):
        widget.setEnabled(enabled)
