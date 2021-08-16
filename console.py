from datetime import datetime

import PyQt5.QtCore as qtc

from config import APP_CONFIG

from enums import BODY_KEY, CONSOLE_COLORS, CONSOLE_MESSAGE_TYPE

from message.message import Message

console_message_template = '<span style="font-size:{}pt; ' + \
    'color:{}; white-space:pre;">{}<span>'


class Console(qtc.QObject):

    __instance = None
    _int_print_sig = qtc.pyqtSignal(Message)
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
    def print(msg: Message):
        Console.get_instance()._int_print_sig.emit(msg)

    @qtc.pyqtSlot(Message)
    def _print(self, msg: Message):
        data = msg.body.data
        msg_type = data[BODY_KEY.CONSOLE_MESSAGE_TYPE]
        msg = data[BODY_KEY.MESSAGE]
        msg_type_prefix = self._get_console_message_prefix(msg_type)
        curr_time_prefix = self._get_current_time_prefix() + ' - '
        text = msg_type_prefix + \
            console_message_template.format(
                APP_CONFIG.app.gui.widgets.console.text_size,
                CONSOLE_COLORS.WHITE.value,
                curr_time_prefix + msg
            )
        self.print_sig.emit(text)

    @staticmethod
    def _get_current_time_prefix():
        return '[' + datetime.now().strftime('%H:%M:%S') + ']'

    @staticmethod
    def _get_console_message_prefix(message_type: CONSOLE_MESSAGE_TYPE):
        prefix_color = message_type.value.prefix_color.value
        prefix = message_type.value.prefix
        prefix = console_message_template.format(
            APP_CONFIG.app.gui.widgets.console.text_size,
            prefix_color,
            f'{prefix: <11}',
        )
        return prefix
