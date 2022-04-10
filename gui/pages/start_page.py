from typing import Dict, Optional

import PyQt5.QtCore as qtc

from enums import SIGNAL_OWNER
from gui.pages.page import Page
from gui.templates.start_page import Ui_start_page
from names import (
    DETECT_DEEPFAKE_PAGE_NAME,
    START_PAGE_NAME,
    MAKE_DEEPFAKE_PAGE_NAME,
    START_PAGE_TITLE,
)


class StartPage(Page, Ui_start_page):

    def __init__(
        self,
        signals: Optional[Dict[SIGNAL_OWNER, qtc.pyqtSignal]] = None,
    ):
        super().__init__(signals, START_PAGE_NAME)

        self.setupUi(self)
        self.setWindowTitle(START_PAGE_TITLE)
        self.make_deepfake_btn.clicked.connect(
            lambda: self._goto(MAKE_DEEPFAKE_PAGE_NAME)
        )
        self.detect_deepfake_btn.clicked.connect(
            lambda: self._goto(DETECT_DEEPFAKE_PAGE_NAME)
        )

    def _goto(self, page):
        self.show_toolbars_and_console(True)
        self.goto(page)
