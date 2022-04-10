import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from config import APP_CONFIG
from console import Console
from enums import (
    APP_STATUS,
    BODY_KEY,
    SIGNAL_OWNER,
    WIDGET,
)
from gui.pages.detect_deepfake_page.detect_deepfake_page import \
    DetectDeepFakePage
from gui.widgets.common import VerticalSpacer, HorizontalSpacer
from gui.widgets.job_info_window import JobInfoWindow
from gui.pages.make_deepfake_page.make_deepfake_page import MakeDeepfakePage
from gui.pages.page import Page
from gui.pages.start_page import StartPage
from gui.templates.main_page import Ui_main_page
from gui.workers.threads.frames_extraction_worker_thread \
    import FramesExtractionWorkerThread
from gui.workers.threads.io_worker_thread import IO_WorkerThread
from gui.workers.threads.message_worker_thread import MessageWorkerThread
from gui.workers.threads.next_element_worker_thread \
    import NextElementWorkerThread
from message.message import Message
from names import MAKE_DEEPFAKE_PAGE_NAME, START_PAGE_NAME


class MainPage(qwt.QMainWindow, Ui_main_page):

    # -- gui signals ---
    show_menubar_sig = qtc.pyqtSignal(bool)
    show_console_sig = qtc.pyqtSignal(bool)
    show_toolbar_sig = qtc.pyqtSignal(bool)
    job_progress_sig = qtc.pyqtSignal(Message)
    app_status_label_sig = qtc.pyqtSignal(str)
    configure_widget_sig = qtc.pyqtSignal(Message)
    job_progressbar_value_sig = qtc.pyqtSignal(int)

    # -- worker signals ---
    io_worker_sig = qtc.pyqtSignal(Message)
    message_worker_sig = qtc.pyqtSignal(Message)
    next_element_worker_sig = qtc.pyqtSignal(Message)
    frames_extraction_worker_sig = qtc.pyqtSignal(Message)

    # -- next element signals ---
    frames_extraction_worker_next_element_sig = qtc.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.show_menubar_sig.connect(self.show_menubar)
        self.show_console_sig.connect(self.show_console)
        self.show_toolbar_sig.connect(self.show_toolbar)
        self.job_progress_sig.connect(self.job_progress)
        self.configure_widget_sig.connect(self.configure_widget)

        Console.get_instance().print_sig.connect(self._console_print)

        self.job_info_window = JobInfoWindow(self)

        # -- setup workers --
        self.setup_io_worker()
        # self.setup_frame_extraction_worker()
        # self.setup_next_element_worker()
        self.setup_message_worker()

        self.m_pages = {}

        self.init_ui()

        self.goto(START_PAGE_NAME)

        # self.goto(MAKE_DEEPFAKE_PAGE_NAME)

    def init_ui(self):
        self.setupUi(self)

        self.register_pages()
        self.init_console()
        self.init_menubar()
        self.init_toolbar()
        self.init_statusbar()

        self.resize(
            APP_CONFIG.app.gui.window.preferred_width,
            APP_CONFIG.app.gui.window.preferred_height,
        )

    def init_toolbar(self):
        self.toolbar = qwt.QToolBar(self)
        self.addToolBar(qtc.Qt.LeftToolBarArea, self.toolbar)
        self.toolbar.setToolButtonStyle(qtc.Qt.ToolButtonTextBesideIcon)
        icon = qwt.QApplication.style().standardIcon(qwt.QStyle.SP_ArrowLeft)
        back_action = self.toolbar.addAction(icon, 'Back')
        back_action.triggered.connect(self.test)

        icon = qtg.QIcon(qtg.QPixmap(':/job_info.svg'))
        job_info = self.toolbar.addAction(icon, 'Job info')
        job_info.triggered.connect(self.open_job_info)
        self.show_toolbar(False)

    def test(self):
        print('ok')

    def terminate_threads(self):
        print('terminating')

    def open_job_info(self):
        self.job_info_window.show()

    def init_console(self):
        font = qtg.QFont(APP_CONFIG.app.gui.widgets.console.font_name)
        self.console.setFont(font)
        self.show_console(False)
        p = self.console.viewport().palette()
        p.setColor(self.console.viewport().backgroundRole(),
                   qtg.QColor(109, 107, 106))
        self.console.viewport().setPalette(p)

    def init_menubar(self):
        file_menu = self.menubar.addMenu('File')
        file_menu.addAction('Settings', self.settings)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close)

        self.menubar.addMenu('Help')

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
        io_worker_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.message_worker_sig,
        }
        self.io_worker_thread = IO_WorkerThread(
            self.io_worker_sig,
            io_worker_signals,
        )
        self.io_worker_thread.start()

    def setup_message_worker(self):
        message_worker_signals = {
            SIGNAL_OWNER.IO_WORKER: self.io_worker_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER:
            self.frames_extraction_worker_sig,
            SIGNAL_OWNER.NEXT_ELEMENT_WORKER: self.next_element_worker_sig,
            SIGNAL_OWNER.JOB_PROGRESS: self.job_progress_sig,
            SIGNAL_OWNER.CONFIGURE_WIDGET: self.configure_widget_sig,
        }
        self.message_worker_thread = MessageWorkerThread(
            self.message_worker_sig,
            message_worker_signals,
        )
        self.message_worker_thread.start()

    def setup_frame_extraction_worker(self):
        frames_extraction_worker_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.message_worker_sig,
        }
        self.frames_extraction_worker_thread = FramesExtractionWorkerThread(
            self.frames_extraction_worker_sig,
            frames_extraction_worker_signals,
            self.frames_extraction_worker_next_element_sig,
        )
        self.frames_extraction_worker_thread.start()

    def setup_next_element_worker(self):
        next_element_worker_signals = {
            SIGNAL_OWNER.MESSAGE_WORKER: self.message_worker_sig,
            SIGNAL_OWNER.FRAMES_EXTRACTION_WORKER_NEXT_ELEMENT:
            self.frames_extraction_worker_next_element_sig,
        }
        self.next_element_worker_thread = NextElementWorkerThread(
            self.next_element_worker_sig,
            next_element_worker_signals,
        )
        self.next_element_worker_thread.start()

    def configure_widget(self, msg: Message):
        data = msg.body.data
        widget = data[BODY_KEY.WIDGET]
        widget_method = data[BODY_KEY.METHOD]
        method_args = data[BODY_KEY.ARGS]
        if widget == WIDGET.JOB_PROGRESS:
            method = getattr(self.job_progressbar, widget_method)
            method(*method_args)

    def settings(self):
        """App settings window.
        """
        self.settings_window = qwt.QMainWindow(self)
        self.settings_window.setWindowTitle('Settings')
        self.settings_window.setFixedSize(500, 400)

        central_wgt = qwt.QWidget()
        central_wgt_layout = qwt.QVBoxLayout()
        central_wgt.setLayout(central_wgt_layout)

        tab_wgt = qwt.QTabWidget(self.settings_window)
        central_wgt_layout.addWidget(tab_wgt)

        # -- device tab ---
        device_tab_wgt = qwt.QWidget(tab_wgt)
        device_tab_wgt_layout = qwt.QVBoxLayout(device_tab_wgt)
        device_tab_wgt.setLayout(device_tab_wgt_layout)
        tab_wgt.addTab(device_tab_wgt, 'Device')

        device_row = qwt.QWidget()
        device_row_layout = qwt.QHBoxLayout()
        device_row.setLayout(device_row_layout)

        device_row_layout.addWidget(qwt.QLabel(text='Device'))

        self.devices_dropdown = qwt.QComboBox()
        for device in APP_CONFIG.app.core.devices:
            self.devices_dropdown.addItem(device.value, device)
        device_row_layout.addWidget(self.devices_dropdown)

        device_tab_wgt_layout.addWidget(device_row)
        device_tab_wgt_layout.addItem(VerticalSpacer)

        # -- window tab ---
        window_tab_wgt = qwt.QWidget(tab_wgt)
        window_tab_wgt_layout = qwt.QVBoxLayout()
        window_tab_wgt.setLayout(window_tab_wgt_layout)

        preferred_width_row = qwt.QWidget()
        preferred_width_row_layout = qwt.QHBoxLayout()
        preferred_width_row.setLayout(preferred_width_row_layout)
        preferred_width_row_layout.addWidget(
            qwt.QLabel(text='Preferred width'),
        )
        self.preferred_width_edit = qwt.QLineEdit()
        self.preferred_width_edit.setText(
            str(APP_CONFIG.app.gui.window.preferred_width),
        )
        preferred_width_row_layout.addWidget(self.preferred_width_edit)

        preferred_height_row = qwt.QWidget()
        preferred_height_row_layout = qwt.QHBoxLayout()
        preferred_height_row.setLayout(preferred_height_row_layout)
        preferred_height_row_layout.addWidget(
            qwt.QLabel(text='Preferred height'))
        self.preferred_height_edit = qwt.QLineEdit()
        self.preferred_height_edit.setText(
            str(APP_CONFIG.app.gui.window.preferred_height),
        )
        preferred_height_row_layout.addWidget(self.preferred_height_edit)

        window_tab_wgt_layout.addWidget(preferred_width_row)
        window_tab_wgt_layout.addWidget(preferred_height_row)
        window_tab_wgt_layout.addItem(VerticalSpacer)

        tab_wgt.addTab(window_tab_wgt, 'Window')

        # -- bottom buttons --
        button_row = qwt.QWidget()
        button_row_layout = qwt.QHBoxLayout()
        button_row.setLayout(button_row_layout)
        central_wgt_layout.addWidget(button_row)
        button_row_layout.addItem(HorizontalSpacer)

        ok_btn = qwt.QPushButton(text='Ok')
        ok_btn.setFixedWidth(120)
        ok_btn.clicked.connect(self.save_settings)
        button_row_layout.addWidget(ok_btn)

        cancel_btn = qwt.QPushButton(text='Cancel')
        cancel_btn.setFixedWidth(120)
        cancel_btn.clicked.connect(self.settings_window.close)
        button_row_layout.addWidget(cancel_btn)

        self.settings_window.setCentralWidget(central_wgt)
        self.settings_window.show()

    def save_settings(self):
        """Updates current app setting with new ones.
        """
        selected_device = self.devices_dropdown.currentData()
        APP_CONFIG.app.core.selected_device = selected_device
        self.settings_window.close()

    def register_page(self, page: Page):
        self.m_pages[page.page_name] = page
        self.stacked_widget.addWidget(page)
        if isinstance(page, Page):
            page.goto_sig.connect(self.goto)

    def register_pages(self):
        for page in [StartPage, MakeDeepfakePage, DetectDeepFakePage]:
            page_signals = {
                SIGNAL_OWNER.MESSAGE_WORKER: self.message_worker_sig,
                SIGNAL_OWNER.SHOW_CONSOLE: self.show_console_sig,
                SIGNAL_OWNER.SHOW_MENUBAR: self.show_menubar_sig,
                SIGNAL_OWNER.SHOW_TOOLBAR: self.show_toolbar_sig,
            }
            p = page(page_signals)
            self.register_page(p)

    @qtc.pyqtSlot(Message)
    def job_progress(self, msg: Message):
        if self.job_progress_value == 0:
            self.show_widget(self.job_progressbar, True)
            self.app_status_label_sig.emit(
                APP_STATUS.BUSY.value +
                f' - {msg.body.data.get(BODY_KEY.JOB_NAME, "")}'
            )

        self.job_progress_value += 1
        self.job_progressbar_value_sig.emit(self.job_progress_value)

        finished = msg.body.finished
        if finished:
            self.show_widget(self.job_progressbar, False)
            self.app_status_label_sig.emit(APP_STATUS.NO_JOB.value)
            self.job_progress_value = 0

    @qtc.pyqtSlot(bool)
    def show_console(self, show: bool):
        self.show_widget(self.console, show)

    @qtc.pyqtSlot(bool)
    def show_menubar(self, show: bool):
        self.show_widget(self.menubar, show)

    @qtc.pyqtSlot(bool)
    def show_toolbar(self, show: bool):
        self.show_widget(self.toolbar, show)

    @qtc.pyqtSlot(str)
    def _console_print(self, msg: str):
        self.console.append(msg)

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
