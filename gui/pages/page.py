from typing import Dict, Optional

import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER
from gui.widgets.base_widget import BaseWidget
from message.message import Message


class Page(BaseWidget):

    goto_sig = qtc.pyqtSignal(str)

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
        page_name='page',
    ) -> None:
        super().__init__(signals)

        self.page_name = page_name

    @property
    def message_worker_sig(self) -> qtc.pyqtSignal:
        return self.signals[SIGNAL_OWNER.MESSAGE_WORKER]

    def send_message(self, msg: Message) -> None:
        self.message_worker_sig.emit(msg)

    def show_menubar(self, show: bool) -> None:
        self.signals[SIGNAL_OWNER.SHOW_MENUBAR].emit(show)

    def show_console(self, show: bool) -> None:
        self.signals[SIGNAL_OWNER.SHOW_CONSOLE].emit(show)

    def show_toolbar(self, show: bool) -> None:
        self.signals[SIGNAL_OWNER.SHOW_TOOLBAR].emit(show)

    def show_toolbars_and_console(self, show: bool) -> None:
        self.show_menubar(show)
        self.show_toolbar(show)
        self.show_console(show)

    def goto(self, name):
        self.goto_sig.emit(name)
