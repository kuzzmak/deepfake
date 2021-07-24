from gui.workers.threads.frames_extraction_worker_thread \
    import FramesExtractionWorkerThread
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.pages.start_page import StartPage
from gui.pages.make_deepfake_page import MakeDeepfakePage

from gui.templates.main_page import Ui_main_page

from gui.workers.threads.io_worker_thread import IO_WorkerThread
from gui.workers.threads.message_worker_thread import MessageWorkerThread

from message.message import Message

from constants import (
    CONSOLE_FONT_NAME,
    CONSOLE_TEXT_SIZE,
    PREFERRED_HEIGHT,
    PREFERRED_WIDTH,
)

from enums import (
    CONSOLE_COLORS,
    CONSOLE_MESSAGE_TYPE,
    SIGNAL_OWNER,
)

from names import START_PAGE_NAME

console_message_template = '<span style="font-size:{}pt; color:{}; white-space:pre;">{}<span>'


class MainPage(qwt.QMainWindow, Ui_main_page):

    # -- gui signals ---
    show_menubar_sig = qtc.pyqtSignal(bool)
    show_console_sig = qtc.pyqtSignal(bool)
    show_toolbar_sig = qtc.pyqtSignal(bool)
    console_print_sig = qtc.pyqtSignal(Message)

    # -- worker signals ---
    frame_extraction_worker_sig = qtc.pyqtSignal(Message)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_menubar_sig.connect(self.show_menubar)
        self.show_console_sig.connect(self.show_console)
        self.show_toolbar_sig.connect(self.show_toolbar)
        self.console_print_sig.connect(self.console_print)

        # -- setup workers --
        self.setup_io_worker()
        self.setup_frame_extraction_worker()
        self.setup_message_worker()

        self.m_pages = {}

        self.init_ui()

        self.goto(START_PAGE_NAME)

    def init_ui(self):
        self.setupUi(self)

        font = qtg.QFont(CONSOLE_FONT_NAME)
        self.console.setFont(font)
        self.show_console(False)

        self.register_pages()
        self.init_menubar()
        self.init_toolbar()

        self.resize(PREFERRED_WIDTH, PREFERRED_HEIGHT)

    def init_toolbar(self):
        self.toolbar = qwt.QToolBar(self)
        self.addToolBar(qtc.Qt.LeftToolBarArea, self.toolbar)
        self.toolbar.setToolButtonStyle(qtc.Qt.ToolButtonTextBesideIcon)
        icon = qwt.QApplication.style().standardIcon(qwt.QStyle.SP_ArrowLeft)
        self.toolbar.addAction(icon, 'back')
        self.show_toolbar(False)

    def init_menubar(self):
        file_menu = self.menubar.addMenu('File')
        file_menu.addAction('Settings', self.settings)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close)

        help_menu = self.menubar.addMenu('Help')

        self.show_menubar(False)

    def setup_io_worker(self):
        self.io_worker_thread = IO_WorkerThread()
        self.io_worker_thread.start()

    def setup_message_worker(self):
        self.message_worker_thread = MessageWorkerThread()
        self.message_worker_thread.worker.add_signal(
            self.console_print_sig, SIGNAL_OWNER.CONOSLE)
        self.message_worker_thread.worker.add_signal(
            self.frame_extraction_worker_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER
        )
        self.message_worker_thread.start()

    def setup_frame_extraction_worker(self):
        self.frame_extraction_worker_thread = FramesExtractionWorkerThread()
        self.frame_extraction_worker_sig.connect(
            self.frame_extraction_worker_thread.worker.process)
        self.frame_extraction_worker_thread.start()

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
            page.goto_sig.connect(self.goto)

    def register_pages(self):
        self.register_page(StartPage(self))
        self.register_page(MakeDeepfakePage(self))

    @qtc.pyqtSlot(bool)
    def show_console(self, show: bool):
        self.show_widget(self.console, show)

    @qtc.pyqtSlot(bool)
    def show_menubar(self, show: bool):
        self.show_widget(self.menubar, show)

    @qtc.pyqtSlot(bool)
    def show_toolbar(self, show: bool):
        self.show_widget(self.toolbar, show)

    @qtc.pyqtSlot(Message)
    def console_print(self, message: Message):
        msg_type, msg = message.body.get_data()
        prefix = self._get_console_message_prefix(msg_type)
        text = prefix + \
            console_message_template.format(
                CONSOLE_TEXT_SIZE, CONSOLE_COLORS.BLACK.value, msg)
        self.console.append(text)

    @qtc.pyqtSlot(str)
    def goto(self, name: str):
        if name in self.m_pages:
            page = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(page)
            self.setWindowTitle(page.windowTitle())

    def show_widget(self, widget: qwt.QWidget, show: bool):
        if show:
            widget.show()
        else:
            widget.hide()

    @staticmethod
    def _get_console_message_prefix(message_type: CONSOLE_MESSAGE_TYPE):
        prefix_color = message_type.value.prefix_color.value
        prefix = message_type.value.prefix
        prefix = console_message_template.format(
            CONSOLE_TEXT_SIZE, prefix_color, f'{prefix: <11}')
        return prefix
