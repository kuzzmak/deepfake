import PyQt6.QtCore as qtc

from config import APP_CONFIG
from enums import (
    COLOR,
    Level,
    LevelColor,
)

console_message_template = '<span style="font-size:{}pt; ' + \
    'color:{}; white-space:pre;">{}<span>'


class Console(qtc.QObject):

    __instance = None
    _int_print_sig = qtc.pyqtSignal(str, str, Level, str)
    print_sig = qtc.pyqtSignal(str)

    def __init__(self):
        if Console.__instance is not None:
            raise Exception('This class shouldn\'t be manually initialized.')
        else:
            super().__init__()
            Console.__instance = self
            self._int_print_sig.connect(self._print)

    @staticmethod
    def get_instance():
        if Console.__instance is None:
            Console()
        return Console.__instance

    @staticmethod
    def print(date: str, name: str, level: Level, msg: str):
        Console.get_instance()._int_print_sig.emit(date, name, level, msg)

    @qtc.pyqtSlot(str, str, Level, str)
    def _print(self, date: str, name: str, level: Level, msg: str):
        prefix = self._get_prefix(date, name, level)
        msg = console_message_template.format(
            APP_CONFIG.app.gui.widgets.console.text_size,
            COLOR.WHITE.value,
            msg
        )
        self.print_sig.emit(prefix + msg)

    @staticmethod
    def _get_prefix(date: str, name: str, level: Level):
        prefix = console_message_template.format(
            APP_CONFIG.app.gui.widgets.console.text_size,
            LevelColor[level.value].value.value,
            f'[{date}] - {name} - {level.value} - '
        )
        return prefix
