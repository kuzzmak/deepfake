import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.templates.main_page import Ui_main_page
from gui.pages.start_page import StartPage
from gui.pages.make_deepfake_page import MakeDeepfakePage

from constants import PREFERRED_HEIGHT, PREFERRED_WIDTH

from names import MAKE_DEEPFAKE_PAGE_NAME, START_PAGE_NAME, START_PAGE_TITLE


class AppPage(qwt.QMainWindow, Ui_main_page):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)

        self.m_pages = {}
        self.register_pages()

        # self.setGeometry(0, 0, MAX_WIDTH, MAX_HEIGHT)
        # self.setMaximumHeight(MAX_HEIGHT)
        # self.setMaximumWidth(MAX_WIDTH)
        self.resize(PREFERRED_WIDTH, PREFERRED_HEIGHT)
        self.setWindowTitle(START_PAGE_TITLE)

        self.goto(START_PAGE_NAME)

    def goto_make_deepfake(self):
        self.goto(MAKE_DEEPFAKE_PAGE_NAME)

    def register_page(self, page: Page):
        self.m_pages[page.page_name] = page
        self.stacked_widget.addWidget(page)
        if isinstance(page, Page):
            page.gotoSignal.connect(self.goto)

    def register_pages(self):
        self.register_page(StartPage())
        self.register_page(MakeDeepfakePage())

    @qtc.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            page = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(page)
            self.setWindowTitle(page.windowTitle())
