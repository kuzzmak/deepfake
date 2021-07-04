from PyQt5 import QtCore
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QLabel, QPushButton, QSplitter, QTextEdit, QVBoxLayout, QWidget

from gui.pages.page import Page
from gui.templates.make_deepfake_page import Ui_make_deepfake_page

from names import MAKE_DEEPFAKE_PAGE_NAME, MAKE_DEEPFAKE_PAGE_TITLE, START_PAGE_NAME


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)
        self.name = MAKE_DEEPFAKE_PAGE_NAME

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.back_btn.clicked.connect(self.goto_start_page)

        self.pushButton.clicked.connect(self.write)

    def goto_start_page(self):
        self.goto(START_PAGE_NAME)

    def write(self):
        self.console.moveCursor(QTextCursor.End)
        self.console.insertPlainText('kurac baho moj' + '\n')
        self.console.moveCursor(QTextCursor.End)
