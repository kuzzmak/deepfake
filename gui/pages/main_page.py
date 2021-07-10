import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.templates.main_page import Ui_main_page
from gui.pages.start_page import StartPage
from gui.pages.make_deepfake_page import MakeDeepfakePage

from constants import PREFERRED_HEIGHT, PREFERRED_WIDTH

from names import MAKE_DEEPFAKE_PAGE_NAME, START_PAGE_NAME, START_PAGE_TITLE


class MainPage(qwt.QMainWindow, Ui_main_page):

    show_menu_bar_sig = qtc.pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_menu_bar_sig.connect(self.show_menu_bar)
        self.setupUi(self)

        self.m_pages = {}
        self.register_pages()
        self.init_toolbar()

        self.resize(PREFERRED_WIDTH, PREFERRED_HEIGHT)
        
        self.goto(START_PAGE_NAME)

    def init_toolbar(self):
        self.menubar = self.menuBar()
        file_menu = self.menubar.addMenu('File')
        file_menu.addAction('Settings', self.settings)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close)

        help_menu = self.menubar.addMenu('Help')

        self.show_menu_bar_sig.emit(False)

    def settings(self):
        settings_window = qwt.QMainWindow(self)
        cw = qwt.QWidget()
        layout = qwt.QVBoxLayout()
        btn_ok = qwt.QPushButton(text='ok')
        layout.addWidget(btn_ok)
        cw.setLayout(layout)
        settings_window.setCentralWidget(cw)
        settings_window.show()

    def register_page(self, page: Page):
        self.m_pages[page.page_name] = page
        self.stacked_widget.addWidget(page)
        if isinstance(page, Page):
            page.gotoSignal.connect(self.goto)

    def register_pages(self):
        self.register_page(StartPage(self))
        self.register_page(MakeDeepfakePage(self))

    def show_menu_bar(self, show: bool):
        if show:
            self.menubar.show()
        else:
            self.menubar.hide()

    @qtc.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            page = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(page)
            self.setWindowTitle(page.windowTitle())
