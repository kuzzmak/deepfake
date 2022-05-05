import logging
from queue import LifoQueue
from typing import Dict

import PyQt6.QtGui as qtg
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qwt

from config import APP_CONFIG
from configs.mri_gan_config import (
    MRIGANConfig,
    generate_default_mri_gan_config,
)
from console import Console
from enums import (
    APP_STATUS,
    BODY_KEY,
    MESSAGE_TYPE,
    SIGNAL_OWNER,
    WORKER_THREAD,
    WIDGET,
)
from gui.pages.detect_deepfake_page.detect_deepfake_page import \
    DetectDeepFakePage
from gui.widgets.common import (
    Button,
    HWidget,
    VWidget,
    VerticalSpacer,
    HorizontalSpacer,
)
from gui.widgets.job_info_window import JobInfoWindow
from gui.pages.make_deepfake_page.make_deepfake_page import MakeDeepfakePage
from gui.pages.page import Page
from gui.pages.start_page import StartPage
from gui.templates.main_page import Ui_main_page
from gui.workers.threads.io_worker_thread import IO_WorkerThread
from gui.workers.threads.message_worker_thread import MessageWorkerThread
from message.message import Message
from names import MAKE_DEEPFAKE_PAGE_NAME, START_PAGE_NAME
from variables import ETA_FORMAT, MRI_GAN_CONFIG_PATH

logger = logging.getLogger(__name__)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._threads: Dict[WORKER_THREAD, qtc.QThread] = dict()

        self.show_menubar_sig.connect(self.show_menubar)
        self.show_console_sig.connect(self.show_console)
        self.show_toolbar_sig.connect(self.show_toolbar)
        self.job_progress_sig.connect(self.job_progress)
        self.configure_widget_sig.connect(self.configure_widget)

        Console.get_instance().print_sig.connect(self._console_print)

        self.job_info_window = JobInfoWindow(self)

        # -- setup workers --
        # self.setup_io_worker()
        self.setup_message_worker()

        self.m_pages = {}
        self._page_nav = LifoQueue()

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
        self.addToolBar(qtc.Qt.ToolBarArea.LeftToolBarArea, self.toolbar)
        self.toolbar.setToolButtonStyle(
            qtc.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        icon = qwt.QApplication.style().standardIcon(
            qwt.QStyle.StandardPixmap.SP_ArrowLeft
        )
        back_action = self.toolbar.addAction(icon, 'Back')
        back_action.triggered.connect(self._go_back)

        icon = qwt.QApplication.style().standardIcon(
            qwt.QStyle.StandardPixmap.SP_FileDialogInfoView
        )
        job_info = self.toolbar.addAction(icon, 'Job info')
        job_info.triggered.connect(self.open_job_info)
        self.show_toolbar(False)

    def _go_back(self):
        _ = self._page_nav.get_nowait()
        page = self._page_nav.queue[-1]
        self.goto(page, False)

    def terminate_threads(self):
        """Closes running threads gracefully before exiting application.
        """
        for k, thread in self._threads.items():
            thread.quit()
            thread.wait()

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

        self.eta_label = qwt.QLabel()
        self.statusbar.addWidget(self.eta_label)

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
            SIGNAL_OWNER.JOB_PROGRESS: self.job_progress_sig,
            SIGNAL_OWNER.CONFIGURE_WIDGET: self.configure_widget_sig,
        }
        message_worker_thread = MessageWorkerThread(
            self.message_worker_sig,
            message_worker_signals,
        )
        message_worker_thread.start()
        self._threads[WORKER_THREAD.MESSAGE_WORKER] = message_worker_thread

    def configure_widget(self, msg: Message):
        data = msg.body.data
        widget = data[BODY_KEY.WIDGET]
        widget_method = data[BODY_KEY.METHOD]
        method_args = data[BODY_KEY.ARGS]
        if widget == WIDGET.JOB_PROGRESS:
            method = getattr(self.job_progressbar, widget_method)
            method(*method_args)
            self.show_widget(self.job_progressbar, True)
            self.app_status_label_sig.emit(
                APP_STATUS.BUSY.value +
                f' - {msg.body.data.get(BODY_KEY.JOB_NAME, "")}'
            )
            self.job_progressbar_value_sig.emit(0)

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
        device_tab_wgt_layout.addItem(VerticalSpacer())

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
        window_tab_wgt_layout.addItem(VerticalSpacer())

        tab_wgt.addTab(window_tab_wgt, 'Window')

        # --- configs tab ---
        configs_tab = VWidget()
        tab_wgt.addTab(configs_tab, 'Configs')

        mri_gan_config = VWidget()
        configs_tab.layout().addWidget(mri_gan_config)

        mri_gan_config.layout().addWidget(qwt.QLabel(
            text='MRI GAN config DFDC base path'
        ))

        base_path = HWidget()
        mri_gan_config.layout().addWidget(base_path)
        base_path.layout().setContentsMargins(0, 0, 0, 0)

        self.mri_gan_config_base_path_input = qwt.QLineEdit()
        base_path.layout().addWidget(self.mri_gan_config_base_path_input)

        select_base_path_btn = Button(text='select')
        base_path.layout().addWidget(select_base_path_btn)
        select_base_path_btn.clicked.connect(
            self._select_mri_gan_config_base_path
        )

        recreate_mri_gan_config_btn = Button(
            text='Recreate config'
        )
        mri_gan_config.layout().addWidget(recreate_mri_gan_config_btn)
        recreate_mri_gan_config_btn.clicked.connect(
            self._recreate_mri_gan_config
        )

        configs_tab.layout().addItem(VerticalSpacer())

        # -- bottom buttons --
        button_row = qwt.QWidget()
        button_row_layout = qwt.QHBoxLayout()
        button_row.setLayout(button_row_layout)
        central_wgt_layout.addWidget(button_row)
        button_row_layout.addItem(HorizontalSpacer())

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

    @qtc.pyqtSlot()
    def _recreate_mri_gan_config(self) -> None:
        """Recreates MRI GAN config with the newly set up base path.
        """
        selected_dir = self.mri_gan_config_base_path_input.text()
        if not selected_dir:
            logger.error(
                'No directory was provided for the MRI GAN config base path.'
            )
            return
        generate_default_mri_gan_config(selected_dir)
        MRIGANConfig.invalidate_instance()
        logger.info(
            'Generated new MRI GAN config with the base ' +
            f'{selected_dir} located in {str(MRI_GAN_CONFIG_PATH)}'
        )

    @qtc.pyqtSlot()
    def _select_mri_gan_config_base_path(self) -> None:
        """Selects directory for the base of the MRI GAN config.
        """
        selected_dir = str(qwt.QFileDialog.getExistingDirectory(
            self,
            'Select directory path',
        ))
        if not selected_dir:
            logger.warning('No directory selected.')
            return
        logger.debug(
            f'Selected {selected_dir} for the base of the MRI GAN config.'
        )
        self.mri_gan_config_base_path_input.setText(selected_dir)

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

    def _finish_job(self) -> None:
        """Removes training status label and progress bar.
        """
        self.show_widget(self.job_progressbar, False)
        self.app_status_label_sig.emit(APP_STATUS.NO_JOB.value)
        self.job_progress_value = 0
        self.eta_label.setText('')

    @qtc.pyqtSlot(Message)
    def job_progress(self, msg: Message):
        if msg.type == MESSAGE_TYPE.JOB_EXIT:
            self._finish_job()
            return

        self.job_progress_value += 1
        self.job_progressbar_value_sig.emit(self.job_progress_value)
        eta = msg.body.data.get(BODY_KEY.ETA, None)
        if eta is not None:
            self.eta_label.setText(ETA_FORMAT.format(eta))

        if msg.body.finished:
            self._finish_job()

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
    def goto(self, name: str, add_to_nav: bool = True):
        page = self.m_pages[name]
        self.stacked_widget.setCurrentWidget(page)
        self.setWindowTitle(page.windowTitle())
        if add_to_nav:
            self._page_nav.put(name)
        if name == START_PAGE_NAME:
            self.show_console(False)
            self.show_toolbar(False)
            self.show_menubar(False)

    def show_widget(self, widget: qwt.QWidget, show: bool):
        if show:
            widget.show()
        else:
            widget.hide()
