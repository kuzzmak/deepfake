from datetime import datetime
from gui.workers.threads.face_detection_worker_thread \
    import FaceDetectionWorkerThread

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from config import APP_CONFIG

from gui.pages.page import Page
from gui.pages.start_page import StartPage
from gui.pages.make_deepfake_page.make_deepfake_page import MakeDeepfakePage

from gui.templates.main_page import Ui_main_page

from gui.workers.threads.frames_extraction_worker_thread \
    import FramesExtractionWorkerThread
from gui.workers.threads.io_worker_thread import IO_WorkerThread
from gui.workers.threads.message_worker_thread import MessageWorkerThread

from message.message import ConsolePrintMessageBody, Message

from enums import (
    APP_STATUS,
    CONSOLE_COLORS,
    CONSOLE_MESSAGE_TYPE,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WIDGET,
)

from names import START_PAGE_NAME

console_message_template = '<span style="font-size:{}pt; ' + \
    'color:{}; white-space:pre;">{}<span>'


class MainPage(qwt.QMainWindow, Ui_main_page):

    # -- gui signals ---
    show_menubar_sig = qtc.pyqtSignal(bool)
    show_console_sig = qtc.pyqtSignal(bool)
    show_toolbar_sig = qtc.pyqtSignal(bool)
    console_print_sig = qtc.pyqtSignal(Message)
    job_progress_sig = qtc.pyqtSignal(Message)
    job_progressbar_value_sig = qtc.pyqtSignal(int)
    app_status_label_sig = qtc.pyqtSignal(str)
    configure_widget_sig = qtc.pyqtSignal(Message)

    # -- worker signals ---
    message_worker_sig = qtc.pyqtSignal(Message)
    frame_extraction_worker_sig = qtc.pyqtSignal(Message)
    io_worker_sig = qtc.pyqtSignal(Message)
    face_detection_worker_sig = qtc.pyqtSignal(Message)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_menubar_sig.connect(self.show_menubar)
        self.show_console_sig.connect(self.show_console)
        self.show_toolbar_sig.connect(self.show_toolbar)
        self.console_print_sig.connect(self.console_print)
        self.job_progress_sig.connect(self.job_progress)
        self.configure_widget_sig.connect(self.configure_widget)

        # -- setup workers --
        self.setup_io_worker()
        self.setup_frame_extraction_worker()
        self.setup_face_detection_worker()
        self.setup_message_worker()

        self.m_pages = {}

        self.init_ui()

        self.goto(START_PAGE_NAME)

    def init_ui(self):
        self.setupUi(self)

        font = qtg.QFont(APP_CONFIG.app.console.font_name)
        self.console.setFont(font)
        self.show_console(False)

        self.register_pages()
        self.init_menubar()
        self.init_toolbar()
        self.init_statusbar()

        self.resize(APP_CONFIG.app.window.preferred_width,
                    APP_CONFIG.app.window.preferred_height)

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

    def init_statusbar(self):
        self.statusbar.addWidget(qwt.QLabel(text='Status: '))
        self.app_status_label = qwt.QLabel(self, text=APP_STATUS.NO_JOB.value)
        self.app_status_label_sig.connect(self.app_status_label.setText)
        self.statusbar.addWidget(self.app_status_label)

        self.job_progressbar = qwt.QProgressBar(self)
        self.job_progressbar_value_sig.connect(self.job_progressbar.setValue)
        self.job_progressbar.setMinimum(0)
        self.job_progressbar.setFormat(' %v/%m (%p%)')

        self.job_progress_value = 0

        self.statusbar.addWidget(self.job_progressbar)

        self.show_widget(self.job_progressbar, False)

    def setup_io_worker(self):
        self.io_worker_thread = IO_WorkerThread()
        self.io_worker_sig.connect(self.io_worker_thread.worker.process)
        self.io_worker_thread.start()

    def setup_message_worker(self):
        self.message_worker_thread = MessageWorkerThread()
        self.message_worker_thread.worker.add_signal(
            self.console_print_sig,
            SIGNAL_OWNER.CONOSLE,
        )
        self.message_worker_thread.worker.add_signal(
            self.io_worker_sig,
            SIGNAL_OWNER.IO_WORKER,
        )
        self.message_worker_thread.worker.add_signal(
            self.frame_extraction_worker_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER,
        )
        self.message_worker_thread.worker.add_signal(
            self.configure_widget_sig,
            SIGNAL_OWNER.CONFIGURE_WIDGET,
        )
        self.message_worker_thread.worker.add_signal(
            self.job_progress_sig,
            SIGNAL_OWNER.JOB_PROGRESS,
        )
        self.message_worker_thread.worker.add_signal(
            self.face_detection_worker_sig,
            SIGNAL_OWNER.FACE_DETECTION_WORKER
        )
        self.message_worker_sig.connect(
            self.message_worker_thread.worker.process)
        self.message_worker_thread.start()

    def setup_frame_extraction_worker(self):
        self.frame_extraction_worker_thread = FramesExtractionWorkerThread()
        self.frame_extraction_worker_thread.worker.add_signal(
            self.message_worker_sig,
            SIGNAL_OWNER.MESSAGE_WORKER
        )
        self.frame_extraction_worker_sig.connect(
            self.frame_extraction_worker_thread.worker.process)
        self.frame_extraction_worker_thread.start()

    def setup_face_detection_worker(self):
        self.face_detection_worker_thread = FaceDetectionWorkerThread()
        self.face_detection_worker_thread.worker.add_signal(
            self.message_worker_sig,
            SIGNAL_OWNER.MESSAGE_WORKER
        )
        self.face_detection_worker_sig.connect(
            self.face_detection_worker_thread.worker.process
        )
        self.face_detection_worker_thread.start()

    def configure_widget(self, msg: Message):
        widget, widget_method, method_args = msg.body.get_data()
        if widget == WIDGET.JOB_PROGRESS:
            method = getattr(self.job_progressbar, widget_method)
            method(*method_args)

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
        for page in [StartPage, MakeDeepfakePage]:
            p = page(self)
            p.add_signal(self.console_print_sig, SIGNAL_OWNER.CONOSLE)
            self.register_page(p)

    @qtc.pyqtSlot(Message)
    def job_progress(self, msg: Message):

        _, finished = msg.body.get_data()

        if self.job_progress_value == 0:
            self.show_widget(self.job_progressbar, True)
            self.app_status_label_sig.emit(APP_STATUS.BUSY.value)

        self.job_progress_value += 1
        self.job_progressbar_value_sig.emit(self.job_progress_value)

        if finished:
            self.show_widget(self.job_progressbar, False)
            self.app_status_label_sig.emit(APP_STATUS.NO_JOB.value)
            self.job_progress_value = 0

            msg = Message(
                MESSAGE_TYPE.REQUEST,
                ConsolePrintMessageBody(
                    CONSOLE_MESSAGE_TYPE.LOG,
                    'Frames extraction finished.'
                )
            )

            self.console_print_sig.emit(msg)

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
        msg_type_prefix = self._get_console_message_prefix(msg_type)
        curr_time_prefix = '[' + datetime.now().strftime('%H:%M:%S') + '] - '
        text = msg_type_prefix + \
            console_message_template.format(
                APP_CONFIG.app.console.text_size,
                CONSOLE_COLORS.BLACK.value,
                curr_time_prefix + msg)
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
            APP_CONFIG.app.console.text_size, prefix_color, f'{prefix: <11}')
        return prefix
