from enum import Enum
from collections import namedtuple

import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qwt

from gui.pages.page import Page
from gui.widgets.video_player import VideoPlayer
from gui.templates.make_deepfake_page import Ui_make_deepfake_page

from names import MAKE_DEEPFAKE_PAGE_NAME, MAKE_DEEPFAKE_PAGE_TITLE, START_PAGE_NAME
from constants import CONSOLE_FONT_NAME, CONSOLE_TEXT_SIZE


ConsolePrefix = namedtuple('ConsolePrefix', 'prefix prefix_color')

console_message_template = '<span style="font-size:{}pt; color:{}; white-space:pre;">{}<span>'


class CONSOLE_COLORS(Enum):
    RED = '#ff0000'
    BLACK = '#000000'
    ORANGE = '#ffa500'


class CONSOLE_MESSAGE_TYPE(Enum):
    LOG = ConsolePrefix('[LOG]', CONSOLE_COLORS.BLACK)
    INFO = ConsolePrefix('[INFO]', CONSOLE_COLORS.BLACK)
    ERROR = ConsolePrefix('[ERROR]', CONSOLE_COLORS.RED)
    WARNING = ConsolePrefix('[WARNING]', CONSOLE_COLORS.ORANGE)


class MakeDeepfakePage(Page, Ui_make_deepfake_page):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setupUi(self)
        self.page_name = MAKE_DEEPFAKE_PAGE_NAME

        self.setWindowTitle(MAKE_DEEPFAKE_PAGE_TITLE)

        self.back_btn.clicked.connect(self.goto_start_page)

        self.pushButton.clicked.connect(self.console_print)

        self.select_video_btn.clicked.connect(self.select_video)

        self.tabs.setTabEnabled(1, False)  # disable tab 2
        self.tabs.setTabEnabled(2, False)  # disable tab 3

        font = qtg.QFont(CONSOLE_FONT_NAME)
        self.console.setFont(font)

        self.video_player = VideoPlayer()
        self.tab_1_layout.addWidget(self.video_player)
        self.tab_1_layout.addWidget(qwt.QLabel(text='labels'))
        self.video_player.hide()

    def goto_start_page(self):
        self.goto(START_PAGE_NAME)

    def _get_console_message_prefix(self, message_type: CONSOLE_MESSAGE_TYPE):
        prefix_color = message_type.value.prefix_color.value
        prefix = message_type.value.prefix
        prefix = console_message_template.format(
            CONSOLE_TEXT_SIZE, prefix_color, f'{prefix: <11}')
        return prefix

    def console_print(self, message: str, message_type: CONSOLE_MESSAGE_TYPE):
        prefix = self._get_console_message_prefix(message_type)
        text = prefix + \
            console_message_template.format(
                CONSOLE_TEXT_SIZE, CONSOLE_COLORS.BLACK.value, message)
        self.console.append(text)

    def select_video(self):
        options = qwt.QFileDialog.Options()
        options |= qwt.QFileDialog.DontUseNativeDialog
        fileName, _ = qwt.QFileDialog.getOpenFileName(
            self, 'Select video file', "data/videos", "Video files (*.mp4)", options=options)
        if fileName:
            self.video_player.video_selection.emit(fileName)
            self.video_player.show()
            self.console_print('Loaded video: ' + fileName, CONSOLE_MESSAGE_TYPE.ERROR)
        else:
            self.console_print('No video selected', CONSOLE_MESSAGE_TYPE.WARNING)

        self.console_print('No video selected', CONSOLE_MESSAGE_TYPE.LOG)
        self.console_print('No video selected', CONSOLE_MESSAGE_TYPE.INFO)
