import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.start_page import StartPage
from gui.pages.make_deepfake_page import MakeDeepfakePage
from gui.pages.page import Page

from names import APP_NAME
from constants import MAX_HEIGHT, MAX_WIDTH, PREFERRED_HEIGHT, PREFERRED_WIDTH


class App(qwt.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stacked_widget = qwt.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}
        self.register_pages()

        # self.setGeometry(0, 0, MAX_WIDTH, MAX_HEIGHT)
        # self.setMaximumHeight(MAX_HEIGHT)
        # self.setMaximumWidth(MAX_WIDTH)
        self.resize(PREFERRED_WIDTH, PREFERRED_HEIGHT)
        self.setWindowTitle(APP_NAME)

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
